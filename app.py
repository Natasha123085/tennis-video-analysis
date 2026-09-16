from website import create_app
app = create_app()
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
if __name__ == '__main__':
    app.run(debug=True)