from flask import Flask, render_template, request
from recipe_search import boolean_search, rank_recipes

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recipes = []
    if request.method == 'POST':
        query = request.form['query']
        filtered = boolean_search(query)
        ranked = rank_recipes(filtered, query)
        recipes = ranked[['name', 'ingredients', 'description']].head(25).to_dict(orient='records')
    return render_template('index.html', recipes=recipes)

if __name__ == '__main__':
    app.run(debug=True)
