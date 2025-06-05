/**
 * A very direct script to force tags to display inline
 */
document.addEventListener('DOMContentLoaded', function() {
  // Apply this after everything else
  setTimeout(function() {
    // Select all links that look like tag links
    var tagLinks = document.querySelectorAll('a[href*="#"]');
    
    // Force each tag link to display inline
    for (var i = 0; i < tagLinks.length; i++) {
      var link = tagLinks[i];
      
      // Check if this is really a tag link
      if (link.getAttribute('href').includes('/tags/') || 
          link.getAttribute('href').startsWith('#')) {
        
        // Force inline display
        link.setAttribute('style', 'display: inline !important');
        
        // Fix the parent too if it exists
        if (link.parentNode) {
          link.parentNode.setAttribute('style', 'display: inline !important');
          
          // And fix any siblings that might be commas
          var siblings = link.parentNode.childNodes;
          for (var j = 0; j < siblings.length; j++) {
            var sibling = siblings[j];
            if (sibling.nodeType === 3 && sibling.textContent.includes(',')) {
              // This is a text node with a comma
              var span = document.createElement('span');
              span.setAttribute('style', 'display: inline !important');
              span.textContent = sibling.textContent;
              sibling.parentNode.replaceChild(span, sibling);
            }
          }
        }
      }
    }
  }, 500);  // Delay to make sure the page is fully loaded
});
