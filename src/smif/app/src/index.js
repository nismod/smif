import 'babel-polyfill'
import React from 'react'
import { render } from 'react-dom'
import { Provider } from 'react-redux'
import { Route, BrowserRouter as Router } from 'react-router-dom'

import Nav from './components/Nav'
import Welcome from './components/Welcome'
import ProjectOverview from './containers/Configuration/Overview/ProjectOverview'
import SosModelRunConfig from './containers/Configuration/Forms/SosModelRunConfig'
import SosModelConfig from './containers/Configuration/Forms/SosModelConfig'
import SectorModelConfig from './containers/Configuration/Forms/SectorModelConfig'
import ScenarioSetConfig from './containers/Configuration/Forms/ScenarioSetConfig'
import NarrativeSetConfig from './containers/Configuration/Forms/NarrativeSetConfig'
import NarrativeConfig from './containers/Configuration/Forms/NarrativeConfig'

import store from './store/store.js'

import 'bootstrap/dist/css/bootstrap.min.css'
import '../static/css/main.css'

render(
    <Provider store={store}>
        <Router>
            <div className="container-fluid row">
                    <Route path="/" component={Nav}/>
                <main role="main" className="col-md-9 ml-sm-auto col-lg-10 pt-3 px-4">
                <div>
                    <Route exact path="/" component={Welcome}/>
                    <Route exact strict path="/configure/:name" component={ProjectOverview}/>
                    <Route exact strict path="/configure/sos-model-run/:name" component={SosModelRunConfig}/>
                    <Route exact strict path="/configure/sos-models/:name" component={SosModelConfig}/>
                    <Route exact strict path="/configure/sector-models/:name" component={SectorModelConfig}/>
                    <Route exact strict path="/configure/scenario-set/:name" component={ScenarioSetConfig}/>
                    <Route exact strict path="/configure/narrative-set/:name" component={NarrativeSetConfig}/>
                    <Route exact strict path="/configure/narratives/:name" component={NarrativeConfig}/>
                </div>
                </main>
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
)
