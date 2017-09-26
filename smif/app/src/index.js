import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/app.jsx';
import 'normalize.css';
import '../static/css/main.css';

ReactDOM.render(
    <App heading="Welcome to smif" />,
    document.getElementById('root')
);
