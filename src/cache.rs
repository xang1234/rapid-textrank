//! Intelligent content-addressable LRU cache for extraction results.
//!
//! Provides a thread-safe, opt-in cache that stores results keyed by a hash
//! of the input text (and any variant-specific parameters such as focus terms
//! or topic weights). When a document is extracted a second time with the same
//! extractor configuration, the cached result is returned instantly — no
//! tokenization, graph construction, or PageRank needed.
//!
//! # Design
//!
//! * **Content-addressable** — keys are 64-bit hashes computed from the raw
//!   text plus any call-site parameters (focus terms, topic weights, etc.).
//! * **LRU eviction** — bounded capacity; least-recently-used entries are
//!   evicted first when the cache is full.
//! * **Thread-safe** — uses `parking_lot::RwLock` for low-contention
//!   concurrent access across Python threads.
//! * **Opt-in** — disabled by default (zero overhead). Enable with
//!   `enable_cache(capacity)`.

use parking_lot::RwLock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ============================================================================
// Cache statistics
// ============================================================================

/// Snapshot of cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum capacity.
    pub capacity: usize,
}

impl CacheStats {
    /// Hit-rate as a fraction in `[0.0, 1.0]`. Returns `0.0` when no lookups
    /// have been performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ============================================================================
// LRU cache (internal)
// ============================================================================

struct CacheEntry<V> {
    key: u64,
    value: V,
}

/// A simple Vec-backed LRU cache.
///
/// Entries are ordered oldest-first. On access an entry is moved to the back
/// (most-recently-used position). Eviction removes from the front (LRU).
struct LruCache<V: Clone> {
    entries: Vec<CacheEntry<V>>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl<V: Clone> LruCache<V> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(1024)),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: u64) -> Option<V> {
        if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            self.hits += 1;
            let entry = self.entries.remove(pos);
            let value = entry.value.clone();
            self.entries.push(entry);
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: u64, value: V) {
        // Remove existing entry with same key (update).
        self.entries.retain(|e| e.key != key);
        // Evict LRU if at capacity.
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(CacheEntry { key, value });
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len(),
            capacity: self.capacity,
        }
    }
}

// ============================================================================
// Thread-safe extraction cache
// ============================================================================

/// A thread-safe, opt-in LRU cache for extraction results.
///
/// Created in the *disabled* state. Call [`enable`](Self::enable) to start
/// caching, or construct with [`with_capacity`](Self::with_capacity) to start
/// enabled.
pub struct ExtractionCache<V: Clone> {
    inner: RwLock<Option<LruCache<V>>>,
}

impl<V: Clone> ExtractionCache<V> {
    /// Create a cache in the disabled state (zero overhead).
    pub fn disabled() -> Self {
        Self {
            inner: RwLock::new(None),
        }
    }

    /// Create a cache that is immediately enabled with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(Some(LruCache::new(capacity))),
        }
    }

    /// Enable the cache with the given maximum capacity.
    ///
    /// If the cache was already enabled with a different capacity, it is
    /// cleared and re-created.
    pub fn enable(&self, capacity: usize) {
        let mut guard = self.inner.write();
        if guard.as_ref().map_or(true, |c| c.capacity != capacity) {
            *guard = Some(LruCache::new(capacity));
        }
    }

    /// Disable the cache, dropping all entries.
    pub fn disable(&self) {
        *self.inner.write() = None;
    }

    /// Returns `true` if caching is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.inner.read().is_some()
    }

    /// Look up a cached value by key.
    pub fn get(&self, key: u64) -> Option<V> {
        self.inner.write().as_mut()?.get(key)
    }

    /// Insert a value into the cache.
    pub fn insert(&self, key: u64, value: V) {
        if let Some(cache) = self.inner.write().as_mut() {
            cache.insert(key, value);
        }
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&self) {
        if let Some(cache) = self.inner.write().as_mut() {
            cache.clear();
        }
    }

    /// Return a snapshot of cache statistics, or `None` if disabled.
    pub fn stats(&self) -> Option<CacheStats> {
        self.inner.read().as_ref().map(|c| c.stats())
    }
}

impl<V: Clone> Default for ExtractionCache<V> {
    fn default() -> Self {
        Self::disabled()
    }
}

// ============================================================================
// Key helpers
// ============================================================================

/// Compute a 64-bit cache key from text alone.
pub fn hash_text(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Compute a 64-bit cache key from text plus additional hashable parameters.
pub fn hash_text_with_params<H: Hash>(text: &str, params: &H) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    params.hash(&mut hasher);
    hasher.finish()
}

/// Convenience: wrap `ExtractionCache` in an `Arc` for sharing across threads.
pub fn shared_cache<V: Clone>() -> Arc<ExtractionCache<V>> {
    Arc::new(ExtractionCache::disabled())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic_hit_miss() {
        let mut cache = LruCache::<String>::new(2);
        cache.insert(1, "one".into());
        cache.insert(2, "two".into());

        assert_eq!(cache.get(1), Some("one".into()));
        assert_eq!(cache.get(3), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.capacity, 2);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::<i32>::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30); // evicts key=1

        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some(20));
        assert_eq!(cache.get(3), Some(30));
    }

    #[test]
    fn test_lru_access_refreshes_order() {
        let mut cache = LruCache::<i32>::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.get(1); // refresh key=1; now key=2 is LRU
        cache.insert(3, 30); // evicts key=2 (LRU)

        assert_eq!(cache.get(1), Some(10)); // still present
        assert_eq!(cache.get(2), None); // evicted
        assert_eq!(cache.get(3), Some(30));
    }

    #[test]
    fn test_lru_update_existing() {
        let mut cache = LruCache::<i32>::new(2);
        cache.insert(1, 10);
        cache.insert(1, 100); // update
        assert_eq!(cache.get(1), Some(100));
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn test_lru_clear() {
        let mut cache = LruCache::<i32>::new(4);
        cache.insert(1, 10);
        cache.insert(2, 20);
        let _ = cache.get(1);
        cache.clear();

        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_extraction_cache_disabled_by_default() {
        let cache = ExtractionCache::<i32>::disabled();
        assert!(!cache.is_enabled());
        assert_eq!(cache.get(42), None);
        assert!(cache.stats().is_none());
    }

    #[test]
    fn test_extraction_cache_enable_disable() {
        let cache = ExtractionCache::<String>::disabled();
        cache.enable(16);
        assert!(cache.is_enabled());

        cache.insert(hash_text("hello"), "world".into());
        assert_eq!(cache.get(hash_text("hello")), Some("world".into()));

        cache.disable();
        assert!(!cache.is_enabled());
        assert_eq!(cache.get(hash_text("hello")), None);
    }

    #[test]
    fn test_extraction_cache_with_capacity() {
        let cache = ExtractionCache::<i32>::with_capacity(8);
        assert!(cache.is_enabled());

        let stats = cache.stats().unwrap();
        assert_eq!(stats.capacity, 8);
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_hash_text_deterministic() {
        let a = hash_text("the quick brown fox");
        let b = hash_text("the quick brown fox");
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_text_different_inputs() {
        let a = hash_text("hello");
        let b = hash_text("world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_text_with_params() {
        let a = hash_text_with_params("hello", &vec!["focus1"]);
        let b = hash_text_with_params("hello", &vec!["focus2"]);
        let c = hash_text_with_params("hello", &vec!["focus1"]);
        assert_ne!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 3,
            misses: 1,
            entries: 2,
            capacity: 4,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shared_cache() {
        let cache = shared_cache::<String>();
        assert!(!cache.is_enabled());
        cache.enable(4);
        let cache2 = cache.clone();
        cache.insert(1, "shared".into());
        assert_eq!(cache2.get(1), Some("shared".into()));
    }
}
