import 'babel-polyfill';
import React from 'react';
import { render } from 'react-dom';
import { Provider } from 'react-redux';
import { Route, BrowserRouter as Router } from 'react-router-dom';

import Nav from './components/Nav';
import Welcome from './components/Welcome';
import ProjectOverview from './containers/ProjectOverview';
import SosModelRunConfig from './containers/SosModelRunConfig';
import SosModelConfig from './containers/SosModelConfig';

import store from './store/store.js';

import 'normalize.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../static/css/main.css';

render(
    <Provider store={store}>
        <Router>
            <div className="container">
                <Nav />
                <Route exact path="/" component={Welcome}/>
                <Route exact path="/configure" component={ProjectOverview}/>
                <Route path="/configure/sos-model-run/:name" component={SosModelRunConfig}/>
                <Route path="/configure/sos-model/:name" component={SosModelConfig}/>
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
);
