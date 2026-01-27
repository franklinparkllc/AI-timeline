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
    
    // Filter events based on search and category
    function filterEvents() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        const selectedCategory = categoryFilter.value;
        
        let visibleCount = 0;
        
        allEventCards.forEach(card => {
            const eventText = card.textContent.toLowerCase();
            const cardCategory = card.getAttribute('data-category');
            
            const matchesSearch = !searchTerm || eventText.includes(searchTerm);
            const matchesCategory = !selectedCategory || cardCategory === selectedCategory;
            
            if (matchesSearch && matchesCategory) {
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
        
        eventCount.textContent = visibleCount;
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

    // Event listeners
    searchInput.addEventListener('input', filterEvents);
    categoryFilter.addEventListener('change', filterEvents);
    jumpBtn.addEventListener('click', jumpToYear);
    yearJump.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            jumpToYear();
        }
    });
    
    // Initialize
    updateEventCount();
});
