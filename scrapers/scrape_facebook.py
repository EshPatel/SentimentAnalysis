import requests

# Your Facebook Graph API access token
access_token = 'YOUR_ACCESS_TOKEN'

# ID of the Facebook Page you want to access
page_id = 'PAGE_ID'

# Define the endpoint URL to get page posts
posts_url = f'https://graph.facebook.com/v12.0/{page_id}/posts?access_token={access_token}'

# Function to get posts from a page
def get_posts():
    response = requests.get(posts_url)
    if response.status_code == 200:
        posts_data = response.json().get('data', [])
        for post in posts_data:
            post_id = post['id']
            message = post.get('message', 'No message')
            created_time = post['created_time']
            print(f"Post ID: {post_id}")
            print(f"Message: {message}")
            print(f"Created Time: {created_time}")
            print("Getting comments...")
            get_comments(post_id)
            print("-" * 40)
    else:
        print("Error fetching posts:", response.json())

# Function to get comments for a specific post
def get_comments(post_id):
    comments_url = f'https://graph.facebook.com/v12.0/{post_id}/comments?access_token={access_token}'
    response = requests.get(comments_url)
    if response.status_code == 200:
        comments_data = response.json().get('data', [])
        if not comments_data:
            print("No comments available.")
        for comment in comments_data:
            comment_id = comment['id']
            comment_message = comment.get('message', 'No message')
            comment_created_time = comment['created_time']
            print(f"  Comment ID: {comment_id}")
            print(f"  Comment: {comment_message}")
            print(f"  Created Time: {comment_created_time}")
    else:
        print("Error fetching comments:", response.json())

# Run the script to get posts and comments
get_posts()
