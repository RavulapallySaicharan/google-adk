/**
 * Direct fix for tags display - this runs after page load to ensure tags display properly
 */
document.addEventListener('DOMContentLoaded', function() {
  // Find all tag links and make sure they display inline
  var tagLinks = document.querySelectorAll('a[href*="/writing/tags/"]');
  for (var i = 0; i < tagLinks.length; i++) {
    tagLinks[i].style.display = 'inline';
    tagLinks[i].style.whiteSpace = 'normal';
    
    // Also fix any parent containers
    var parent = tagLinks[i].parentNode;
    if (parent) {
      parent.style.display = 'inline';
      parent.style.whiteSpace = 'normal';
      
      var grandparent = parent.parentNode;
      if (grandparent) {
        grandparent.style.display = 'block';
        grandparent.style.whiteSpace = 'normal';
      }
    }
    
    // Fix any commas that might be causing line breaks
    var next = tagLinks[i].nextSibling;
    if (next && next.textContent && next.textContent.includes(',')) {
      next.style.display = 'inline';
      next.style.whiteSpace = 'normal';
    }
  }
  
  // Special fix for .tags-container
  var tagsContainers = document.querySelectorAll('.tags-container');
  for (var j = 0; j < tagsContainers.length; j++) {
    tagsContainers[j].style.display = 'inline';
    tagsContainers[j].style.whiteSpace = 'normal';
  }
});
