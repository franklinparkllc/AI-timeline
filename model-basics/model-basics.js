// Model Basics - Modern Slideshow Story

let currentSlide = 0;
const totalSlides = cardsData.length;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    renderSlides();
    updateNavigation();
    setupEventListeners();
    
    // Check for hash navigation
    const hash = window.location.hash;
    if (hash) {
        const slideIndex = parseInt(hash.replace('#slide-', '')) - 1;
        if (slideIndex >= 0 && slideIndex < totalSlides) {
            currentSlide = slideIndex;
            goToSlide(currentSlide);
        }
    }
});

// Render all slides
function renderSlides() {
    const wrapper = document.getElementById('slidesWrapper');
    
    cardsData.forEach((card, index) => {
        const slide = document.createElement('div');
        slide.className = 'slide';
        slide.setAttribute('data-category', card.category);
        slide.id = `slide-${index + 1}`;
        
        let contentHTML = `
            <div class="slide-content">
                <h1 class="slide-title">${card.title}</h1>
                ${card.description ? `<div class="slide-description">${card.description}</div>` : ''}
                <div class="slide-body">
                    ${card.paragraphs ? card.paragraphs.map(p => `<p>${p}</p>`).join('') : ''}
                    ${card.bullets ? `<ul>${card.bullets.map(bullet => `<li>${bullet}</li>`).join('')}</ul>` : ''}
        `;
        
        // Add callout if exists
        if (card.callout) {
            contentHTML += `
                <div class="callout callout-${card.callout.type}">
                    ${card.callout.content}
                </div>
            `;
        }
        
        // Add resources if exists
        if (card.resources && card.resources.length > 0) {
            contentHTML += `
                <div class="slide-resources">
                    <div class="resources-title">Learn More</div>
                    <div class="resources-grid">
                        ${card.resources.map(res => `
                            <a href="${res.url}" target="_blank" class="resource-link">
                                <span class="resource-icon">${res.icon}</span>
                                <div class="resource-content">
                                    <div class="resource-title">${res.title}</div>
                                    <div class="resource-meta">${res.meta}</div>
                                </div>
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        contentHTML += `
                </div>
            </div>
        `;
        
        slide.innerHTML = contentHTML;
        wrapper.appendChild(slide);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Home button
    document.getElementById('homeBtn').addEventListener('click', () => {
        goToSlide(0);
    });
    
    // Navigation buttons
    document.getElementById('prevBtn').addEventListener('click', previousSlide);
    document.getElementById('nextBtn').addEventListener('click', nextSlide);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            previousSlide();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            nextSlide();
        }
    });
    
    // Touch swipe support
    let touchStartX = 0;
    let touchEndX = 0;
    
    const wrapper = document.getElementById('slidesWrapper');
    
    wrapper.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    wrapper.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                nextSlide();
            } else {
                previousSlide();
            }
        }
    }
}

// Navigation functions
function previousSlide() {
    if (currentSlide > 0) {
        currentSlide--;
        goToSlide(currentSlide);
    }
}

function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        currentSlide++;
        goToSlide(currentSlide);
    }
}

function goToSlide(index) {
    currentSlide = index;
    const wrapper = document.getElementById('slidesWrapper');
    wrapper.style.transform = `translateX(-${currentSlide * 100}%)`;
    
    // Update URL hash
    window.location.hash = `slide-${currentSlide + 1}`;
    
    updateNavigation();
}

function updateNavigation() {
    // Update progress
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const progress = ((currentSlide + 1) / totalSlides) * 100;
    
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${currentSlide + 1} / ${totalSlides}`;
    
    // Update button states
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    prevBtn.disabled = currentSlide === 0;
    nextBtn.disabled = currentSlide === totalSlides - 1;
}

// Handle hash changes (browser back/forward)
window.addEventListener('hashchange', () => {
    const hash = window.location.hash;
    if (hash) {
        const slideIndex = parseInt(hash.replace('#slide-', '')) - 1;
        if (slideIndex >= 0 && slideIndex < totalSlides && slideIndex !== currentSlide) {
            currentSlide = slideIndex;
            goToSlide(currentSlide);
        }
    }
});
