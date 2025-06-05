/**
 * Dark mode functionality
 */
(function() {
  // Function to update the toggle button appearance
  function updateToggleButton(isDark) {
    const toggleButton = document.getElementById('dark-mode-toggle');
    if (!toggleButton) return;
    
    const sunIcon = toggleButton.querySelector('.sun-icon');
    const moonIcon = toggleButton.querySelector('.moon-icon');
    
    if (sunIcon && moonIcon) {
      if (isDark) {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
      } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
      }
    }
  }

  // Apply theme immediately to prevent flash
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
    updateToggleButton(theme === 'dark');
    
    // Toggle logo visibility based on theme
    const lightModeLogo = document.querySelector('.light-mode-logo');
    const darkModeLogo = document.querySelector('.dark-mode-logo');
    
    if (lightModeLogo && darkModeLogo) {
      if (theme === 'dark') {
        lightModeLogo.style.display = 'none';
        darkModeLogo.style.display = 'inline-block';
      } else {
        lightModeLogo.style.display = 'inline-block';
        darkModeLogo.style.display = 'none';
      }
    }
    
    // Update Mermaid theme if Mermaid is loaded
    if (typeof mermaid !== 'undefined') {
      // Update theme
      mermaid.initialize({
        startOnLoad: false,
        theme: theme === 'dark' ? 'dark' : 'default'
      });
      
      try {
        // Clean up the existing diagrams more thoroughly
        document.querySelectorAll('.mermaid').forEach(function(el) {
          // Get the original diagram definition
          var diagramDef = el.getAttribute('data-original-code') || el.textContent || el.innerText;
          
          // Store the original code if not already stored
          if (!el.getAttribute('data-original-code')) {
            el.setAttribute('data-original-code', diagramDef);
          }
          
          // Create a new div to replace the old one
          var newDiv = document.createElement('div');
          newDiv.className = 'mermaid';
          newDiv.setAttribute('data-original-code', diagramDef);
          newDiv.textContent = diagramDef;
          
          // Replace the old element with the new one
          if (el.parentNode) {
            el.parentNode.replaceChild(newDiv, el);
          }
        });
        
        // Re-render all diagrams after a small delay
        setTimeout(function() {
          mermaid.run();
        }, 50);
      } catch (e) {
        console.error('Error updating Mermaid diagrams:', e);
      }
    }
  }

  // Check for saved theme preference or use the system preference
  function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && 
                        window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (savedTheme === null && prefersDark)) {
      applyTheme('dark');
    } else {
      applyTheme('light');
    }
  }

  // Initialize theme right away
  initTheme();
  
  // Set up event listeners when the DOM is ready
  function setupEventListeners() {
    const toggleButton = document.getElementById('dark-mode-toggle');
    
    if (toggleButton) {
      toggleButton.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
      });
    }
    
    // Listen for system preference changes
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        // Only change theme automatically if user hasn't manually set a preference
        if (!localStorage.getItem('theme')) {
          const newTheme = e.matches ? 'dark' : 'light';
          applyTheme(newTheme);
        }
      });
    }
  }

  // Set up event listeners when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupEventListeners);
  } else {
    setupEventListeners();
  }
})();