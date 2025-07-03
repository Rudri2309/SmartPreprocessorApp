#!/bin/bash

# 📌 sync.sh - Auto pull, add, commit, push

echo "🔄 Pulling latest from remote..."
git pull origin main --allow-unrelated-histories

echo "➕ Staging all changes..."
git add .

echo "📝 Enter your commit message:"
read commit_msg

git commit -m "$commit_msg"

echo "🚀 Pushing to remote..."
git push origin main

echo "✅ Sync complete!"

