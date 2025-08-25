---
# Basic info
title: "{{ replace .File.BaseFileName "-" " " | title }}"
date: {{ .Date | default now }}
draft: false
description: "Short description of the post."
tags: []
author: "Me" # For multiple authors, use: ["Me", "You"]

# Metadata & SEO
canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

# Table of contents
showToc: true
TocOpen: false
UseHugoToc: true

# Post features
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

# Syntax highlighting
disableHLJS: false # set true to disable Highlight.js
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/CrustularumAmator/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
---
