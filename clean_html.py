from bs4 import BeautifulSoup
import sys
import re

def clean_html_tags(input_file, output_file):
    """
    Clean HTML tags from a text file and Python code blocks while preserving code structure.
    
    Args:
        input_file (str): Path to the input file containing HTML content
        output_file (str): Path to save the cleaned text
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # First, extract and clean Python code blocks
        code_blocks = []
        def save_code_block(match):
            code_content = match.group(1)
            # Clean HTML tags from the code block
            soup = BeautifulSoup(code_content, 'html.parser')
            cleaned_code = soup.get_text()
            # Preserve indentation and newlines
            cleaned_code = re.sub(r'\n\s*\n', '\n', cleaned_code)  # Remove extra blank lines
            cleaned_code = re.sub(r'^\s+', '', cleaned_code, flags=re.MULTILINE)  # Remove leading spaces
            code_blocks.append(cleaned_code)
            return f"CODE_BLOCK_{len(code_blocks)-1}"
        
        # Find and replace Python code blocks with placeholders
        html_content = re.sub(
            r'<pre class="highlight"><code>(.*?)</code></pre>',
            save_code_block,
            html_content,
            flags=re.DOTALL
        )
        
        # Parse HTML and get text
        soup = BeautifulSoup(html_content, 'html.parser')
        cleaned_text = soup.get_text(separator='\n', strip=True)
        
        # Restore Python code blocks
        for i, code in enumerate(code_blocks):
            cleaned_text = cleaned_text.replace(f"CODE_BLOCK_{i}", f"\n\n{code}\n\n")
        
        # Write the cleaned text to output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
            
        print(f"Successfully cleaned HTML tags and preserved code structure. Cleaned content saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_html.py input_file output_file")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        clean_html_tags(input_file, output_file) 