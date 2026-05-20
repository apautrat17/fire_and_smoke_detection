def predict_video(
    model,
    video_path: str,
    conf_threshold: float = 0.5,
    show: bool = True,
    save: bool = False,
):
    model.predict(
        source=video_path,
        conf=conf_threshold,
        show=show,
        save=save,
    )
