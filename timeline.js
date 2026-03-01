/**
 * Interactive Timeline Functionality
 * Handles filtering, searching, scrolling, and navigation for the AI timeline
 */

// Configuration constants
const CONFIG = {
    SCROLL_SPEED_MULTIPLIER: 2,
    SEARCH_DEBOUNCE_MS: 300,
    SCROLL_THROTTLE_MS: 16 // ~60fps
};

/**
 * Debounce function to limit how often a function is called
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function to limit function execution rate
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}


document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const eventCount = document.getElementById('eventCount');
    const timelineWrapper = document.getElementById('timelineWrapper');
    const segButtons = document.querySelectorAll('.seg-btn[data-significance]');
    const timelineScrubber = document.getElementById('timelineScrubber');
    const scrubberYearLabel = document.getElementById('scrubberYearLabel');
    
    // Track current significance level (default = 1, "All")
    let currentSignificance = 1;
    
    // Validate critical elements exist
    if (!timelineWrapper || !searchInput || !categoryFilter || !segButtons.length) {
        console.error('Critical DOM elements missing. Timeline may not function correctly.');
        return;
    }
    
    // Cache event cards and year sections
    const allEventCards = Array.from(document.querySelectorAll('.event-card'));
    const allYearSections = Array.from(document.querySelectorAll('.timeline-year'));
    
    // ============================================
    // UI Controls
    // ============================================

    /**
     * Handle significance segmented button clicks
     */
    function handleSegmentedButtonClick(e) {
        const btn = e.target;
        // Update active state
        segButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update significance level and re-filter
        currentSignificance = parseInt(btn.getAttribute('data-significance'));
        filterEvents();
    }

    segButtons.forEach(btn => {
        btn.addEventListener('click', handleSegmentedButtonClick);
    });
    
    // ============================================
    // Filtering & Search
    // ============================================
    
    /**
     * Update the displayed event count
     */
    function updateEventCount() {
        if (!eventCount) return;
        const visibleCount = allEventCards.filter(card => !card.classList.contains('hidden')).length;
        eventCount.textContent = visibleCount;
    }
    
    /**
     * Filter events based on search term, category, and significance
     */
    function filterEvents() {
        const searchTerm = searchInput?.value.toLowerCase().trim() || '';
        const selectedCategory = categoryFilter?.value || '';
        const minSignificance = currentSignificance;
        
        let visibleCount = 0;
        
        allEventCards.forEach(card => {
            const eventText = card.textContent.toLowerCase();
            const cardCategory = card.getAttribute('data-category');
            const cardWeight = parseInt(card.getAttribute('data-weight') || '0');
            
            const matchesSearch = !searchTerm || eventText.includes(searchTerm);
            const matchesCategory = !selectedCategory || cardCategory === selectedCategory;
            const matchesSignificance = cardWeight >= minSignificance;
            
            if (matchesSearch && matchesCategory && matchesSignificance) {
                card.classList.remove('hidden');
                visibleCount++;
            } else {
                card.classList.add('hidden');
            }
        });
        
        // Hide/show year sections based on visible events
        allYearSections.forEach(yearSection => {
            const yearEvents = yearSection.querySelectorAll('.event-card');
            const hasVisibleEvents = Array.from(yearEvents).some(card => !card.classList.contains('hidden'));
            
            if (hasVisibleEvents) {
                yearSection.style.display = 'flex';
            } else {
                yearSection.style.display = 'none';
            }
        });
        
        updateEventCount();
        
        // Re-sync timeline scrubber after layout may have changed
        requestAnimationFrame(() => {
            syncScrubberFromScroll();
        });
    }
    
    // Debounced search for better performance
    const debouncedFilterEvents = debounce(filterEvents, CONFIG.SEARCH_DEBOUNCE_MS);
    
    // ============================================
    // Drag to Scroll
    // ============================================

    let isDown = false;
    let startX;
    let scrollLeft;

    function handleTimelineMouseDown(e) {
        isDown = true;
        timelineWrapper.classList.add('active');
        startX = e.pageX - timelineWrapper.offsetLeft;
        scrollLeft = timelineWrapper.scrollLeft;
    }

    function handleTimelineMouseLeave() {
        isDown = false;
        timelineWrapper.classList.remove('active');
    }

    function handleTimelineMouseUp() {
        isDown = false;
        timelineWrapper.classList.remove('active');
    }

    function handleTimelineMouseMove(e) {
        if (!isDown) return;
        e.preventDefault();
        const x = e.pageX - timelineWrapper.offsetLeft;
        const walk = (x - startX) * CONFIG.SCROLL_SPEED_MULTIPLIER;
        timelineWrapper.scrollLeft = scrollLeft - walk;
    }

    timelineWrapper.addEventListener('mousedown', handleTimelineMouseDown);
    timelineWrapper.addEventListener('mouseleave', handleTimelineMouseLeave);
    timelineWrapper.addEventListener('mouseup', handleTimelineMouseUp);
    timelineWrapper.addEventListener('mousemove', handleTimelineMouseMove);

    // ============================================
    // Timeline Scrubber & Parallax
    // ============================================
    
    /**
     * Get the maximum scroll position
     * @returns {number} Maximum scroll value
     */
    function getMaxScroll() {
        const max = timelineWrapper.scrollWidth - timelineWrapper.clientWidth;
        return Math.max(0, max);
    }

    /**
     * Sync bottom timeline scrubber with wrapper scroll position (scrubber = 0–100%)
     */
    function syncScrubberFromScroll() {
        if (!timelineScrubber) return;
        const maxScroll = getMaxScroll();
        const pct = maxScroll <= 0 ? 0 : (timelineWrapper.scrollLeft / maxScroll) * 100;
        timelineScrubber.value = Math.min(100, Math.max(0, pct));
    }

    /**
     * Scroll wrapper to position given scrubber value (0–100)
     * @param {number|string} value - Scrubber value (0-100)
     */
    function scrollFromScrubber(value) {
        const maxScroll = getMaxScroll();
        timelineWrapper.scrollLeft = (parseFloat(value) / 100) * maxScroll;
    }

    // Throttled scroll handler for better performance
    function handleTimelineScroll() {
        syncScrubberFromScroll();
    }
    const throttledScrollHandler = throttle(handleTimelineScroll, CONFIG.SCROLL_THROTTLE_MS);

    function handleScrubberInput() {
        scrollFromScrubber(timelineScrubber.value);
    }

    timelineWrapper.addEventListener('scroll', throttledScrollHandler);

    if (timelineScrubber) {
        timelineScrubber.addEventListener('input', handleScrubberInput);
        // Initial sync
        syncScrubberFromScroll();
    }

    // ============================================
    // Event Listeners
    // ============================================

    // Store handlers as named functions so they can be removed during cleanup
    const handleSearchInput = debouncedFilterEvents;
    const handleCategoryChange = filterEvents;

    if (searchInput) {
        searchInput.addEventListener('input', handleSearchInput);
    }

    if (categoryFilter) {
        categoryFilter.addEventListener('change', handleCategoryChange);
    }

    // Significance segmented buttons are already wired up above
    
    // ============================================
    // Scrubber year label
    // ============================================

    // Build sorted array of year sections with their left-offset positions
    function getYearPositions() {
        return Array.from(allYearSections)
            .map(el => ({ year: el.getAttribute('data-year'), left: el.offsetLeft }))
            .sort((a, b) => a.left - b.left);
    }

    function getYearAtScroll(scrollLeft) {
        const positions = getYearPositions();
        if (!positions.length) return '';
        let closest = positions[0];
        for (const p of positions) {
            if (p.left <= scrollLeft + timelineWrapper.clientWidth / 2) closest = p;
            else break;
        }
        return closest.year;
    }

    function updateScrubberYearLabel() {
        if (!scrubberYearLabel || !timelineScrubber) return;
        const pct = parseFloat(timelineScrubber.value) / 100;
        // Position label over thumb: thumb travels from 0 to 100% of input width
        const wrap = timelineScrubber.parentElement;
        const thumbX = pct * timelineScrubber.offsetWidth;
        scrubberYearLabel.style.left = thumbX + 'px';
        scrubberYearLabel.textContent = getYearAtScroll(timelineWrapper.scrollLeft);
    }

    if (timelineScrubber && scrubberYearLabel) {
        function handleScrubberMouseEnter() {
            scrubberYearLabel.classList.add('visible');
            updateScrubberYearLabel();
        }

        function handleScrubberMouseLeave() {
            scrubberYearLabel.classList.remove('visible');
        }

        timelineScrubber.addEventListener('mouseenter', handleScrubberMouseEnter);
        timelineScrubber.addEventListener('mouseleave', handleScrubberMouseLeave);
        timelineScrubber.addEventListener('mousemove', updateScrubberYearLabel);
        timelineScrubber.addEventListener('input', updateScrubberYearLabel);
    }


    // ============================================
    // Initialize
    // ============================================

    filterEvents();

    // ============================================
    // Cleanup on page unload (memory leak prevention)
    // ============================================

    function cleanup() {
        // Remove event listeners to prevent memory leaks
        segButtons.forEach(btn => {
            btn.removeEventListener('click', handleSegmentedButtonClick);
        });

        timelineWrapper.removeEventListener('mousedown', handleTimelineMouseDown);
        timelineWrapper.removeEventListener('mouseleave', handleTimelineMouseLeave);
        timelineWrapper.removeEventListener('mouseup', handleTimelineMouseUp);
        timelineWrapper.removeEventListener('mousemove', handleTimelineMouseMove);
        timelineWrapper.removeEventListener('scroll', throttledScrollHandler);

        if (searchInput) {
            searchInput.removeEventListener('input', handleSearchInput);
        }

        if (categoryFilter) {
            categoryFilter.removeEventListener('change', handleCategoryChange);
        }

        if (timelineScrubber) {
            timelineScrubber.removeEventListener('input', handleScrubberInput);
            timelineScrubber.removeEventListener('mouseenter', handleScrubberMouseEnter);
            timelineScrubber.removeEventListener('mouseleave', handleScrubberMouseLeave);
            timelineScrubber.removeEventListener('mousemove', updateScrubberYearLabel);
            timelineScrubber.removeEventListener('input', updateScrubberYearLabel);
        }
    }

    // Cleanup when page unloads or transitions away (fires before bfcache)
    window.addEventListener('pagehide', cleanup);
});
