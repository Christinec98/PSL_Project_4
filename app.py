# !pip install Dash

# !pip install dash_bootstrap_components

# %%capture
# %run project_4.ipynb

from logging import debug
from typing import Dict

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State
from project_4 import get_displayed_movies, get_recommended_movies

# import time
import os

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], 
               suppress_callback_exceptions=True)
server = app.server

CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), content])


# %%time

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    movies = get_displayed_movies()
    return html.Div(
        [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H1("Rate some movies below to"),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    children=[
                                        "Get recommendations ",
                                        html.I(className="bi bi-emoji-heart-eyes-fill"),
                                    ],
                                    size="lg",
                                    className="btn-success",
                                    id="button-recommend",
                                ),
                                className="p-0",
                            ),
                        ],
                        className="sticky-top bg-white py-2",
                    ),
                    html.Div(
                        [
                            get_movie_card(movie, with_rating=True)
                            for idx, movie in movies.iterrows()
                        ],
                        className="row row-cols-1 row-cols-5",
                        id="rating-movies",
                    ),
                ],
                id="rate-movie-container",
            ),
            html.H1(
                "Your recommendations", id="your-recommendation",  style={"display": "none"}
            ),
            dcc.Loading(
                [
                    dcc.Link(
                        "Try again", href="/system-2", refresh=True, className="mb-2 d-block"
                    ),
                    html.Div(
                        className="row row-cols-1 row-cols-5",
                        id="recommended-movies",
                    ),
                ],
                type="circle",
            ),
        ]
    )

# %%time
def get_movie_card(movie, with_rating=False):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.MovieID}.jpg?raw=true",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.Title, className="card-title text-center"),
                    ]
                ),
            ]
            + (
                [
                    dcc.RadioItems(
                        options=[
                            {"label": "1", "value": "1"},
                            {"label": "2", "value": "2"},
                            {"label": "3", "value": "3"},
                            {"label": "4", "value": "4"},
                            {"label": "5", "value": "5"},
                        ],
                        className="text-center",
                        id={"type": "movie_rating", "movie_id": movie.MovieID},
                        inline=True,
                        inputClassName="m-1",
                        labelClassName="px-1",
                    )
                ]
                if with_rating
                else []
            ),
            className="h-100",
        ),
        className="col mb-4",
    )
    

# %%time
@app.callback(
    Output("rate-movie-container", "style"),
    Output("your-recommendation", "style"),
    [Input("button-recommend", "n_clicks")],
    prevent_initial_call=True,
)    
def on_recommend_button_clicked(n):
    return {"display": "none"}, {"display": "block"}

# %%time
@app.callback(
    Output("recommended-movies", "children"),
    [Input("rate-movie-container", "style")],
    [
        State({"type": "movie_rating", "movie_id": ALL}, "value"),
        State({"type": "movie_rating", "movie_id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)

def on_getting_recommendations(style, ratings, ids):
    rating_input = {
        ids[i]["movie_id"]: int(rating) for i, rating in enumerate(ratings) if rating is not None
    }
    recommended_movies = get_recommended_movies(rating_input)
    return [get_movie_card(movie) for idx, movie in recommended_movies.iterrows()]

# %%time
#if __name__ == "__main__":
#    # app.run_server(port=9000, debug=True)
#    # app.run_server(mode="inline", port=9000, debug=True)
#    app.run_server(mode="external", port=8000, debug=True)




if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)


