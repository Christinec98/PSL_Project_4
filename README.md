# PSL_Project_4

Team Contribution:
*   Syed Ahmed         netID: syeda2    UIN: *****5315 Online MCS
    - Contributions     System I and II
*   Jessica Tomas      netID: jptomas2  UIN: *****0877 Online MCS
    - Contributions     Build the application for System II 
*   Christine Zhou     netID: xizhou4   UIN: *****6213 Online MCS
    - Contributions    Build the application for System II  and deploy the app

Project 4: Movie Recommender System
This project is a movie recommendation system that provides recommendations based on two systems:

Popularity-based Recommendation: Recommends movies with the highest average ratings.
Item-Based Collaborative Filtering (IBCF): Recommends movies based on user preferences and similarity between movies.
The application is built using Dash and deployed on Heroku.



Project Structure:

project-directory/

├── app.py                  # Main application file for Dash

├── project_4.py            # Script for recommendation systems I and II logic

├── Procfile                # Heroku configuration file for Gunicorn

├── requirements.txt        # Python dependencies for the project

├── movie_ratings.csv       # Movie ratings dataset

├── movies.dat              # Movies metadata

├── ratings.dat             # User ratings data

├── gunicorn_config.py      # Optional Gunicorn configuration (if used)

├── MovieImages/            # Directory containing movie images

│   ├── <movie_id>.jpg      # Images for each movie (e.g., 1.jpg, 2.jpg)

└── README.md               # Instructions for setting up and running the project

Setup Instructions:

1. Clone the Repository
   
     $ git clone <repository-url>
     
     $ cd project-directory
   
3. Install Dependencies:  (Use pip to install the required dependencies)
   
     $ pip install -r requirements.txt
   
5. Run the Application Locally
   To test the app locally, use the following command:
   
     $ python app.py
   
   Alternatively, run it with Gunicorn:
   
     $ gunicorn app:server --workers=1 --threads=2 --timeout=120
   
   Open your browser and go to:   http://127.0.0.1:8000
                            or:   http://localhost:8000/system-2

Deploying to Heroku:


1. Create a Heroku App
  If you haven't already, create a Heroku app:

    $ heroku create <app-name>
    
2. Add a Procfile
  Ensure your Procfile contains the following line:

    web: gunicorn app:server --workers=1 --threads=2 --timeout=120
  
3. Push to Heroku
  Deploy the application:

    $ git add .
   
    $ git commit -m "Deploy movie recommendation app"
   
    $ git push heroku main
   
4. Open the App
  Once deployed, open the app in your browser:

    $ heroku open
   
   
Datasets -- The following datasets are included in the project:

    movie_ratings.csv: Matrix of movie ratings by users.
    movies.dat: Metadata about movies (titles, genres).
    ratings.dat: User rating data.
    
Features:
    
    Popularity-Based Recommendation: Displays the top 10 movies based on average ratings.
    IBCF Recommendation: Custom recommendations based on user ratings.
    Interactive Dashboard: Users can interact with the app to rate movies and get personalized recommendations.
    Top Movies Display: Visualization of the top 10 movies with their posters.
    
Directory Descriptions:
    
    app.py: Contains the Dash application and layout.
    project_4.py: Contains the recommendation logic (popularity-based and IBCF systems).
    Procfile: Specifies how to run the app on Heroku.
    requirements.txt: Lists all Python libraries required for the project.
    MovieImages/: Directory containing movie posters.
    
Dependencies:

    dash
    dash-bootstrap-components
    matplotlib
    numpy
    pandas
    gunicorn
    
Install them via:

    $ pip install -r requirements.txt


Acknowledgments -- This app uses datasets from:

    MovieLens: For user ratings and movie metadata:  https://grouplens.org/datasets/movielens/1m/
    
    Movie Posters: Images sourced from external repositories. https://liangfgithub.github.io/MovieData/MovieImages.zip 
    
    Github link: https://github.com/Christinec98/PSL_Project_4
    
    Deployed web address: https://project-4-movie-app-b8887c3d7250.herokuapp.com/

