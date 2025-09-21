// Landing page functionality - FIXED VERSION
function enterApp() {
    console.log('Enter App function called');

    const landingSection = document.getElementById('landing');
    const mainApp = document.getElementById('mainApp');

    if (!landingSection || !mainApp) {
        console.error('Landing section or main app not found');
        // Fallback: just show main app
        document.getElementById('mainApp').style.display = 'block';
        document.getElementById('landing').style.display = 'none';
        initializeApp();
        return;
    }

    // Add exit animation to landing page
    landingSection.style.animation = 'fade-out 0.8s ease-out forwards';

    // Show main app with entrance animation
    setTimeout(() => {
        console.log('Hiding landing page, showing main app');
        landingSection.style.display = 'none';
        mainApp.style.display = 'block';
        mainApp.style.opacity = '0';
        mainApp.style.transform = 'translateY(50px)';

        // Trigger entrance animation
        setTimeout(() => {
            mainApp.style.transition = 'all 0.8s ease-out';
            mainApp.style.opacity = '1';
            mainApp.style.transform = 'translateY(0)';

            // Initialize app after animation
            setTimeout(() => {
                initializeApp();
            }, 800);
        }, 100);
    }, 800);
}

// Fallback function in case JavaScript fails
function fallbackEnterApp() {
    console.log('Using fallback enter app function');
    const mainApp = document.getElementById('mainApp');
    const landing = document.getElementById('landing');

    if (mainApp && landing) {
        landing.style.display = 'none';
        mainApp.style.display = 'block';
        initializeApp();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, setting up event listeners');

    // Make sure the button works
    const getStartedBtn = document.getElementById('getStartedBtn');
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Get Started button clicked');
            enterApp();
        });
    }

    // Initialize particles.js for background effect
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: '#ffffff' },
                shape: { type: 'circle' },
                opacity: { value: 0.5, random: false },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#ffffff',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                }
            },
            retina_detect: true
        });
    }
});