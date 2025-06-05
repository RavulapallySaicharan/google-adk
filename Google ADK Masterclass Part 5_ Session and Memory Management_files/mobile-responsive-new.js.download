/**
 * Simple mobile responsiveness enhancements
 */
(function() {
  // Function to run when the DOM is loaded
  function enhanceMobileExperience() {
    // Add aria attributes to improve accessibility
    const mainContent = document.querySelector('main');
    if (mainContent) {
      mainContent.setAttribute('role', 'main');
      mainContent.setAttribute('id', 'main-content');
    }
    
    // Fix for responsive tables
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
      if (!table.parentElement.classList.contains('table-responsive')) {
        const wrapper = document.createElement('div');
        wrapper.classList.add('table-responsive');
        wrapper.style.overflowX = 'auto';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });
    
    // Fix for blog post listing page - simple approach
    if (window.location.pathname.includes('/writing/') || 
        document.querySelector('.posts') || 
        document.querySelector('.post-list')) {
      
      // Direct fix for blog post titles in the listing
      const blogTitles = document.querySelectorAll('.post-title, .posts h1, .posts h2, .post-link');
      blogTitles.forEach(title => {
        // Force proper text wrapping
        title.style.display = 'block';
        title.style.width = '100%';
        title.style.maxWidth = '100%';
        title.style.wordBreak = 'normal';
        title.style.overflowWrap = 'break-word';
        title.style.whiteSpace = 'normal';
        
        // Remove any padding or margins that might interfere with wrapping
        title.style.paddingRight = '0';
        title.style.marginRight = '0';
        title.style.textAlign = 'left';
        
        // Force the text to be visible in mobile browsers
        const computedStyle = window.getComputedStyle(title);
        const textContent = title.textContent;
        
        // If the text is likely to be cut off, try to force it to wrap
        if (textContent.length > 30) {
          // Insert a zero-width space after every few words to encourage wrapping
          const words = textContent.split(' ');
          if (words.length > 3) {
            // We don't actually modify the content, just ensure wrapping
            title.style.wordBreak = 'break-word';
          }
        }
      });
    }
  }
  
  // Initialize when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceMobileExperience);
  } else {
    enhanceMobileExperience();
  }
})();
