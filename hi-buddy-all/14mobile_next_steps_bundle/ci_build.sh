#!/bin/bash
set -e
# CI helper to build web artifacts and copy into capacitor project
# Usage: ./ci_build.sh
echo "Building web assets..."
npm run build || echo "No web build step configured"
echo "Copying into capacitor..."
npx cap copy || true
echo "Done"
