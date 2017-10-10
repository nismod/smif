import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/App.js';
import 'normalize.css';
import '../static/css/main.css';
import './pattern-library.css';

class StyleGuide extends React.Component {
    // 'Hello World' render to style guide page
    // - should be able to visit at http://localhost:8080/style-guide.html
    //   when running the dev server (npm start)
    // - TODO move this component to a suitable directory (containers)
    // - TODO add light-touch style-guide-specific styles
    render() {
        return (
            <article>
                <header className="guide-header">
                    <h1>smif &ndash; Pattern library</h1>
                </header>
                <div className="guide-content">
                    <h2>Components</h2>
                    <div className="sample-container">
                        <App heading="Example Heading" />
                    </div>
                </div>
            </article>);
    }
}

ReactDOM.render(
    <StyleGuide />,
    document.getElementById('root')
);
