import typer

from mediapipe_utils import main

app = typer.Typer(help="Extract pose landmarks and connections from videos.")


@app.command()
def extract(
    in_dir: str = typer.Option(
        "data/raw_videos", help="Input directory containing .mp4 videos"
    ),
    out_dir: str = typer.Option(
        "data/keypoints", help="Output directory for keypoints and connections"
    ),
):
    """Extract pose landmarks and connections from videos."""
    main(in_dir=in_dir, out_dir=out_dir)


if __name__ == "__main__":
    app()
