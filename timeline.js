/**
 * Interactive Timeline Functionality
 * Handles filtering, searching, scrolling, and navigation for the AI timeline
 */

// Configuration constants
const CONFIG = {
    MIN_YEAR: 1940,
    MAX_YEAR: 2025,
    SCROLL_SPEED_MULTIPLIER: 2,
    PARALLAX_MOVEMENT_FACTOR: 150,
    SEARCH_DEBOUNCE_MS: 300,
    YEAR_HIGHLIGHT_DURATION_MS: 1000,
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

/**
 * Show error message to user (replaces alert)
 * @param {string} message - Error message to display
 */
function showError(message) {
    // Create or get error element
    let errorEl = document.getElementById('yearJumpError');
    if (!errorEl) {
        errorEl = document.createElement('div');
        errorEl.id = 'yearJumpError';
        errorEl.className = 'error-message';
        const yearNav = document.querySelector('.year-navigation');
        if (yearNav) {
            yearNav.appendChild(errorEl);
        }
    }
    errorEl.textContent = message;
    errorEl.style.display = 'block';
    
    // Hide after 3 seconds
    setTimeout(() => {
        errorEl.style.display = 'none';
    }, 3000);
}

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const yearJump = document.getElementById('yearJump');
    const jumpBtn = document.getElementById('jumpBtn');
    const eventCount = document.getElementById('eventCount');
    const timelineWrapper = document.getElementById('timelineWrapper');
    const controlsToggle = document.getElementById('controlsToggle');
    const controlsMenu = document.getElementById('controlsMenu');
    const significanceSlider = document.getElementById('significanceSlider');
    const parallaxLogo = document.getElementById('parallaxLogo');
    const timelineScrubber = document.getElementById('timelineScrubber');
    
    // Validate critical elements exist
    if (!timelineWrapper || !searchInput || !categoryFilter || !significanceSlider) {
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
     * Toggle controls menu visibility
     */
    function toggleControlsMenu() {
        if (controlsToggle && controlsMenu) {
            controlsToggle.classList.toggle('active');
            controlsMenu.classList.toggle('active');
        }
    }
    
    if (controlsToggle) {
        controlsToggle.addEventListener('click', toggleControlsMenu);
    }

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (controlsToggle && controlsMenu && 
            !controlsToggle.contains(e.target) && 
            !controlsMenu.contains(e.target)) {
            controlsToggle.classList.remove('active');
            controlsMenu.classList.remove('active');
        }
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
        const minSignificance = parseInt(significanceSlider?.value || '1');
        
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
    // Year Navigation
    // ============================================
    
    /**
     * Jump to a specific year in the timeline
     */
    function jumpToYear() {
        const year = parseInt(yearJump?.value);
        
        if (!year || year < CONFIG.MIN_YEAR || year > CONFIG.MAX_YEAR) {
            showError(`Please enter a valid year between ${CONFIG.MIN_YEAR} and ${CONFIG.MAX_YEAR}`);
            if (yearJump) {
                yearJump.focus();
                yearJump.select();
            }
            return;
        }
        
        // Find the closest year section
        let targetSection = null;
        let minDiff = Infinity;
        
        allYearSections.forEach(section => {
            const sectionYear = parseInt(section.getAttribute('data-year'));
            const diff = Math.abs(sectionYear - year);
            
            if (diff < minDiff) {
                minDiff = diff;
                targetSection = section;
            }
        });
        
        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
            
            // Highlight the year marker briefly
            const yearMarker = targetSection.querySelector('.year-marker');
            if (yearMarker) {
                yearMarker.style.color = 'var(--accent)';
                yearMarker.style.transform = 'translateX(-50%) scale(1.5)';
                yearMarker.style.transition = 'all 0.3s ease';
                
                setTimeout(() => {
                    yearMarker.style.color = 'var(--year-color)';
                    yearMarker.style.transform = 'translateX(-50%) scale(1)';
                }, CONFIG.YEAR_HIGHLIGHT_DURATION_MS);
            }
        }
    }
    
    // ============================================
    // Drag to Scroll
    // ============================================
    
    let isDown = false;
    let startX;
    let scrollLeft;

    timelineWrapper.addEventListener('mousedown', (e) => {
        isDown = true;
        timelineWrapper.classList.add('active');
        startX = e.pageX - timelineWrapper.offsetLeft;
        scrollLeft = timelineWrapper.scrollLeft;
    });
    
    timelineWrapper.addEventListener('mouseleave', () => {
        isDown = false;
        timelineWrapper.classList.remove('active');
    });
    
    timelineWrapper.addEventListener('mouseup', () => {
        isDown = false;
        timelineWrapper.classList.remove('active');
    });
    
    timelineWrapper.addEventListener('mousemove', (e) => {
        if (!isDown) return;
        e.preventDefault();
        const x = e.pageX - timelineWrapper.offsetLeft;
        const walk = (x - startX) * CONFIG.SCROLL_SPEED_MULTIPLIER;
        timelineWrapper.scrollLeft = scrollLeft - walk;
    });

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

    /**
     * Update parallax logo position based on scroll
     */
    function updateParallax() {
        if (!parallaxLogo) return;
        const maxScroll = getMaxScroll();
        const scrollPercentage = maxScroll <= 0 ? 0 : timelineWrapper.scrollLeft / maxScroll;
        const moveX = (scrollPercentage - 0.5) * CONFIG.PARALLAX_MOVEMENT_FACTOR;
        parallaxLogo.style.transform = `translateX(${moveX}px)`;
    }

    // Throttled scroll handler for better performance
    const throttledScrollHandler = throttle(() => {
        syncScrubberFromScroll();
        updateParallax();
    }, CONFIG.SCROLL_THROTTLE_MS);

    timelineWrapper.addEventListener('scroll', throttledScrollHandler);

    if (timelineScrubber) {
        timelineScrubber.addEventListener('input', () => {
            scrollFromScrubber(timelineScrubber.value);
        });
        // Initial sync
        syncScrubberFromScroll();
    }

    // ============================================
    // Accessibility
    // ============================================
    
    /**
     * Update significance slider ARIA attributes for accessibility
     */
    function updateSignificanceAria() {
        if (significanceSlider) {
            significanceSlider.setAttribute('aria-valuenow', significanceSlider.value);
        }
    }
    
    if (significanceSlider) {
        significanceSlider.addEventListener('input', updateSignificanceAria);
    }

    // ============================================
    // Event Listeners
    // ============================================
    
    if (searchInput) {
        searchInput.addEventListener('input', debouncedFilterEvents);
    }
    
    if (categoryFilter) {
        categoryFilter.addEventListener('change', filterEvents);
    }
    
    if (significanceSlider) {
        significanceSlider.addEventListener('input', filterEvents);
    }
    
    if (jumpBtn) {
        jumpBtn.addEventListener('click', jumpToYear);
    }
    
    if (yearJump) {
        yearJump.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                jumpToYear();
            }
        });
    }
    
    // ============================================
    // Initialize
    // ============================================
    
    filterEvents();
});
