import 'babel-polyfill'
import React from 'react'
import { render } from 'react-dom'
import { Provider } from 'react-redux'
import { Route, BrowserRouter as Router } from 'react-router-dom'

import Nav from './components/Nav'
import Welcome from './components/Welcome'
import Footer from './containers/Footer'
import ProjectOverview from './containers/ConfigForm/ProjectOverview'
import SosModelRunConfig from './containers/ConfigForm/SosModelRunConfig'
import SosModelConfig from './containers/ConfigForm/SosModelConfig'
import SectorModelConfig from './containers/ConfigForm/SectorModelConfig'
import ScenarioSetConfig from './containers/ConfigForm/ScenarioSetConfig'
import NarrativeSetConfig from './containers/ConfigForm/NarrativeSetConfig'
import NarrativeConfig from './containers/ConfigForm/NarrativeConfig'

import store from './store/store.js'

import 'bootstrap/dist/css/bootstrap.min.css'
import '../static/css/main.css'

render(
    <Provider store={store}>
        <Router>
            <div className="container">
                <Nav />
                <Route exact path="/" component={Welcome}/>
                <Route exact path="/configure" component={ProjectOverview}/>
                <Route path="/configure/sos-model-run/:name" component={SosModelRunConfig}/>
                <Route path="/configure/sos-models/:name" component={SosModelConfig}/>
                <Route path="/configure/sector-models/:name" component={SectorModelConfig}/>
                <Route path="/configure/scenario-set/:name" component={ScenarioSetConfig}/>
                <Route path="/configure/narrative-set/:name" component={NarrativeSetConfig}/>
                <Route path="/configure/narratives/:name" component={NarrativeConfig}/>
                <Footer />
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
)
