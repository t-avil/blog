# Makefile for Hugo

# Default Hugo command
HUGO=hugo

# Create a new post: make newpost name=your-post-name
newpost:
	@if [ -z "$(name)" ]; then \
		echo "Error: Please provide a post name, e.g., make newpost name=hello-world"; \
		exit 1; \
	fi
	$(HUGO) new --kind post "posts/$(name).md"

# Build the site
build:
	$(HUGO)

# Serve the site locally with live reload
serve:
	$(HUGO) server -D

# Clean Hugo cache
clean:
	$(HUGO) --cleanDestinationDir
