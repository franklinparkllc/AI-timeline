// Interactive Timeline Functionality

document.addEventListener('DOMContentLoaded', function() {
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
    
    // Toggle controls menu
    controlsToggle.addEventListener('click', () => {
        controlsToggle.classList.toggle('active');
        controlsMenu.classList.toggle('active');
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!controlsToggle.contains(e.target) && !controlsMenu.contains(e.target)) {
            controlsToggle.classList.remove('active');
            controlsMenu.classList.remove('active');
        }
    });
    
    let allEventCards = Array.from(document.querySelectorAll('.event-card'));
    let allYearSections = Array.from(document.querySelectorAll('.timeline-year'));
    
    // Update event count
    function updateEventCount() {
        const visibleCount = allEventCards.filter(card => !card.classList.contains('hidden')).length;
        eventCount.textContent = visibleCount;
    }
    
    // Filter events based on search, category and significance
    function filterEvents() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        const selectedCategory = categoryFilter.value;
        const minSignificance = parseInt(significanceSlider.value);
        
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
        
        // Update the event count display
        if (eventCount) {
            eventCount.textContent = visibleCount;
        }
        // Re-sync timeline scrubber after layout may have changed
        requestAnimationFrame(() => {
            if (typeof syncScrubberFromScroll === 'function') syncScrubberFromScroll();
        });
    }
    
    // Jump to specific year
    function jumpToYear() {
        const year = parseInt(yearJump.value);
        if (!year || year < 1940 || year > 2025) {
            alert('Please enter a valid year between 1940 and 2025');
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
            yearMarker.style.color = 'var(--accent)';
            yearMarker.style.transform = 'translateX(-50%) scale(1.5)';
            yearMarker.style.transition = 'all 0.3s ease';
            
            setTimeout(() => {
                yearMarker.style.color = 'var(--year-color)';
                yearMarker.style.transform = 'translateX(-50%) scale(1)';
            }, 1000);
        }
    }
    
    // Drag to scroll functionality
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
        const walk = (x - startX) * 2; // Scroll speed
        timelineWrapper.scrollLeft = scrollLeft - walk;
    });

    // Max scroll position (reusable, updated when needed)
    function getMaxScroll() {
        const max = timelineWrapper.scrollWidth - timelineWrapper.clientWidth;
        return Math.max(0, max);
    }

    // Sync bottom timeline scrubber with wrapper scroll (scrubber = 0–100%)
    function syncScrubberFromScroll() {
        if (!timelineScrubber) return;
        const maxScroll = getMaxScroll();
        const pct = maxScroll <= 0 ? 0 : (timelineWrapper.scrollLeft / maxScroll) * 100;
        timelineScrubber.value = Math.min(100, Math.max(0, pct));
    }

    // Scroll wrapper to position given scrubber value 0–100
    function scrollFromScrubber(value) {
        const maxScroll = getMaxScroll();
        timelineWrapper.scrollLeft = (parseFloat(value) / 100) * maxScroll;
    }

    timelineWrapper.addEventListener('scroll', () => {
        syncScrubberFromScroll();
        const maxScroll = getMaxScroll();
        const scrollPercentage = maxScroll <= 0 ? 0 : timelineWrapper.scrollLeft / maxScroll;
        const moveX = (scrollPercentage - 0.5) * 150;
        if (parallaxLogo) {
            parallaxLogo.style.transform = `translateX(${moveX}px)`;
        }
    });

    if (timelineScrubber) {
        timelineScrubber.addEventListener('input', () => {
            scrollFromScrubber(timelineScrubber.value);
        });
        // Initial sync and sync after layout (e.g. filter changes)
        syncScrubberFromScroll();
    }

    // Update significance slider aria for accessibility
    function updateSignificanceAria() {
        if (significanceSlider) {
            significanceSlider.setAttribute('aria-valuenow', significanceSlider.value);
        }
    }
    significanceSlider.addEventListener('input', updateSignificanceAria);

    // Event listeners
    searchInput.addEventListener('input', filterEvents);
    categoryFilter.addEventListener('change', filterEvents);
    significanceSlider.addEventListener('input', filterEvents);
    jumpBtn.addEventListener('click', jumpToYear);
    yearJump.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            jumpToYear();
        }
    });
    
    // Initialize
    filterEvents();
});
