import React from 'react';
import ReactDOM from 'react-dom';
import App from './hello/app.jsx';
import 'normalize.css';
import '../static/css/main.css';

class StyleGuide extends React.Component {
    // 'Hello World' render to style guide page
    // - should be able to visit at http://localhost:8080/style-guide.html
    //   when running the dev server (npm start)
    // - TODO move this component to a suitable directory (containers)
    // - TODO add light-touch style-guide-specific styles
    render() {
        return (
            <div className="content-wrapper">
                <h1>smif &ndash; Pattern library</h1>
                <h2>Components</h2>
                <App heading="Example Heading" />
            </div>);
    }
}

ReactDOM.render(
    <StyleGuide />,
    document.getElementById('root')
);
