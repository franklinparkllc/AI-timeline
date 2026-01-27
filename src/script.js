// Interactive Timeline Functionality

document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const weightFilter = document.getElementById('weightFilter');
    const yearJump = document.getElementById('yearJump');
    const jumpBtn = document.getElementById('jumpBtn');
    const eventCount = document.getElementById('eventCount');
    const timeline = document.getElementById('timeline');
    
    let allEventCards = Array.from(document.querySelectorAll('.event-card'));
    let allYearSections = Array.from(document.querySelectorAll('.timeline-year'));
    
    // Update event count
    function updateEventCount() {
        const visibleCount = allEventCards.filter(card => !card.classList.contains('hidden')).length;
        eventCount.textContent = visibleCount;
    }
    
    // Filter events based on search, category, and weight
    function filterEvents() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        const selectedCategory = categoryFilter.value;
        const selectedWeight = weightFilter ? weightFilter.value : '';
        
        let visibleCount = 0;
        
        allEventCards.forEach(card => {
            const eventText = card.textContent.toLowerCase();
            const cardCategory = card.getAttribute('data-category');
            const cardWeight = card.getAttribute('data-weight') || '';
            
            const matchesSearch = !searchTerm || eventText.includes(searchTerm);
            const matchesCategory = !selectedCategory || cardCategory === selectedCategory;
            const matchesWeight = !selectedWeight || cardWeight === selectedWeight;
            
            if (matchesSearch && matchesCategory && matchesWeight) {
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
                yearSection.style.display = 'block';
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
            targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Highlight the year marker briefly
            const yearMarker = targetSection.querySelector('.year-marker');
            yearMarker.style.transform = 'scale(1.2)';
            yearMarker.style.transition = 'transform 0.3s ease';
            
            setTimeout(() => {
                yearMarker.style.transform = 'scale(1)';
            }, 500);
        }
    }
    
    // Event listeners
    searchInput.addEventListener('input', filterEvents);
    categoryFilter.addEventListener('change', filterEvents);
    if (weightFilter) {
        weightFilter.addEventListener('change', filterEvents);
    }
    
    jumpBtn.addEventListener('click', jumpToYear);
    yearJump.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            jumpToYear();
        }
    });
    
    // Initialize
    updateEventCount();
    
    // Add smooth scroll behavior for better UX
    const yearMarkers = document.querySelectorAll('.year-marker');
    yearMarkers.forEach(marker => {
        marker.style.cursor = 'pointer';
        marker.addEventListener('click', function() {
            const yearSection = this.closest('.timeline-year');
            yearSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + F focuses search
        if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
            e.preventDefault();
            searchInput.focus();
        }
    });
    
    // Highlight search terms
    function highlightSearchTerms() {
        const searchTerm = searchInput.value.trim();
        if (!searchTerm) {
            // Remove all highlights
            allEventCards.forEach(card => {
                const title = card.querySelector('.event-title');
                if (title) {
                    title.innerHTML = title.textContent;
                }
            });
            return;
        }
        
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        
        allEventCards.forEach(card => {
            const title = card.querySelector('.event-title');
            if (title) {
                const text = title.textContent;
                title.innerHTML = text.replace(regex, '<mark>$1</mark>');
            }
        });
    }
    
    searchInput.addEventListener('input', function() {
        filterEvents();
        highlightSearchTerms();
    });
    
    // Add CSS for mark highlighting
    const style = document.createElement('style');
    style.textContent = `
        mark {
            background-color: #fef08a;
            padding: 2px 4px;
            border-radius: 3px;
        }
    `;
    document.head.appendChild(style);
});
