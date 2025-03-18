#!/usr/bin/env bash
set -e

# Single-handedly the best tool I've ever made.
# Produces the selected files of the repo into a format easily parsable by most LLMs.
# Debuggers be damned!

# --- CONFIGURATION ---

# Directory to scan (default current directory).
FILES_DIR="./"

# Array of file extensions to include.
EXTENSIONS=(".py" ".md")

# Array of filepath contents to exclude.
# If you need more (e.g., "dist", "build", "test"), add them here:
EXCLUSIONS=(".next" "dist" "public" "node_modules" "package-lock" "ios" "android")

# ---------------------

# Initialize a variable to store concatenated file contents.
output=""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to copy to clipboard based on OS
copy_to_clipboard() {
    case "$(uname)" in
        Darwin)
            # macOS
            echo -e "$output" | pbcopy
            ;;
        Linux)
            # Linux (requires xclip or xsel)
            if command_exists xclip; then
                echo -e "$output" | xclip -selection clipboard
            elif command_exists xsel; then
                echo -e "$output" | xsel --clipboard --input
            else
                echo "Error: Install 'xclip' or 'xsel' to enable clipboard functionality on Linux."
                exit 1
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            # Windows (using clip.exe)
            echo -e "$output" | clip.exe
            ;;
        *)
            echo "Unsupported OS: $(uname)"
            exit 1
            ;;
    esac
}

# 1) Build the file-extension matching part of the `find` command
#    e.g. \( -name "*.ts" -o -name "*.tsx" -o ... \)
FIND_EXT_ARGS=()
for ext in "${EXTENSIONS[@]}"; do
    FIND_EXT_ARGS+=( -name "*${ext}" -o )
done
# Remove the trailing "-o" from the last entry
unset 'FIND_EXT_ARGS[${#FIND_EXT_ARGS[@]}-1]'

# 2) Build the exclusion part of the `find` command:
#    For each EXCLUSIONS keyword, we add:  -not -path "*<keyword>*"
#    We also always exclude any path that has node_modules in it.
FIND_EXCLUDE_ARGS=()
# Always exclude node_modules:
FIND_EXCLUDE_ARGS+=( -not -path "*node_modules*" )
# Then exclude anything from the EXCLUSIONS array:
for exclusion in "${EXCLUSIONS[@]}"; do
    FIND_EXCLUDE_ARGS+=( -not -path "*${exclusion}*" )
done

# -- Construct the final find command --
#
# Explanation:
#   -type f                    => Only files
#   ( "${FIND_EXT_ARGS[@]}" )  => File extension matching, with OR logic
#   "${FIND_EXCLUDE_ARGS[@]}"  => Exclude paths containing these patterns
#   -print                     => Print matching file paths
#
# Example in plain English:
#   find . -type f ( -name "*.ts" -o -name "*.tsx" ... ) \
#       -not -path "*node_modules*" \
#       -not -path "*dist*" \
#       -print

# Use a process substitution to read the paths from the find command one by one
while IFS= read -r file; do
    # Detect the file extension
    extension="${file##*.}"

    # Add a comment header before each file's content
    case "$extension" in
        ts|tsx|js|jsx)
            # For script-like files, prepend with //
            output+="// $file\n\n"
            ;;
        html|css)
            # For markup/style files, prepend with /* */
            output+="/* $file */\n\n"
            ;;
        *)
            # Fallback
            output+="# $file\n\n"
            ;;
    esac

    # Append the file content
    file_content=$(cat "$file")
    output+="$file_content\n\n"

done < <(
    # Execute the find command
    find "$FILES_DIR" \
        -type f \( "${FIND_EXT_ARGS[@]}" \) \
        "${FIND_EXCLUDE_ARGS[@]}" \
        -print
)

# Copy all concatenated contents to the clipboard
copy_to_clipboard

# Print to console
echo -e "$output"

# Print success message
echo "All matching file contents have been copied to the clipboard."