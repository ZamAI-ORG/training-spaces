import argparse
from huggingface_hub import snapshot_download, login, upload_folder


def clone_model(repo_id, dest_dir="models"):
    # Clone a model from Hugging Face Hub
    path = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=f"{dest_dir}/{repo_id.replace('/', '_')}")
    print(f"Model downloaded to {path}")


def clone_space(repo_id, dest_dir="spaces"):
    # Clone a space from Hugging Face Hub
    path = snapshot_download(repo_id=repo_id, repo_type="space", local_dir=f"{dest_dir}/{repo_id.replace('/', '_')}")
    print(f"Space downloaded to {path}")


def upload_model_weights(local_dir, repo_id, token=None):
    # Upload model weights to Hugging Face Hub
    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        repo_type="model",
        token=token
    )
    print(f"Uploaded {local_dir} to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Clone Hugging Face models and spaces")
    subparsers = parser.add_subparsers(dest="command")

    # Login to HF Hub
    login_parser = subparsers.add_parser("login", help="Login to Hugging Face Hub")
    login_parser.add_argument("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", help="Hugging Face access token")

    model_parser = subparsers.add_parser("model", help="Clone a model")
    model_parser.add_argument("repo_id", help="The model repo id, e.g. facebook/opt-125m")
    model_parser.add_argument("--dest", default="models", help="Destination directory")

    space_parser = subparsers.add_parser("space", help="Clone a space")
    space_parser.add_argument("repo_id", help="The space repo id, e.g. username/space-name")
    space_parser.add_argument("--dest", default="spaces", help="Destination directory")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload model weights to Hugging Face Hub")
    upload_parser.add_argument("local_dir", help="Local model directory, e.g. models/ZamAI-Mistral-7B-Pashto/")
    upload_parser.add_argument("repo_id", help="Target HF repo id, e.g. tasal9/ZamAI-Mistral-7B-Pashto")
    upload_parser.add_argument("--token", help="Hugging Face access token", default=None)

    args = parser.parse_args()

    # Handle login command
    if args.command == "login":
        login(args.token)
        print("Logged in successfully.")
        return
    
    if args.command == "model":
        clone_model(args.repo_id, dest_dir=args.dest)
    elif args.command == "space":
        clone_space(args.repo_id, dest_dir=args.dest)
    elif args.command == "upload":
        upload_model_weights(args.local_dir, args.repo_id, token=args.token)
        return
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
