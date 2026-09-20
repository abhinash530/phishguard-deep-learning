# Dataset

This directory is reserved for the dataset used by the PhishGuard project.

## Dataset Purpose

The dataset contains examples of URLs or web addresses that can be classified into two categories:

- `0` — Legitimate
- `1` — Phishing

## Expected Dataset Format

A CSV dataset can use the following structure:

```text
url,label
https://example.com,0
http://example-phishing.com/login,1
