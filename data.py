import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

N_TAGS = 40


def load():
    """Retourne titres/labels train et test, et la liste des 40 tags les plus frequents."""
    df = pd.read_csv("stackoverflow_posts.csv")
    top = df.tag_name.value_counts().head(N_TAGS).index
    df = df[df.tag_name.isin(top)]
    posts = df.groupby("post_id").agg(title=("title", "first"), tags=("tag_name", list))

    mlb = MultiLabelBinarizer(classes=list(top))
    Y = mlb.fit_transform(posts.tags)
    return (*train_test_split(posts.title.tolist(), Y, test_size=0.2, random_state=0), mlb.classes_)
