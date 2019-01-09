import '@babel/polyfill'
import React from 'react'
import { render } from 'react-dom'
import { Provider } from 'react-redux'
import { Route, BrowserRouter as Router, Switch } from 'react-router-dom'

import Nav from 'containers/Nav'
import Welcome from 'components/Welcome'
import NotFound from 'components/ConfigForm/General/NotFound'
import JobsOverview from 'containers/Configuration/Overview/JobsOverview'
import JobRunner from 'containers/Simulation/JobRunner'
import ProjectOverview from 'containers/Configuration/Overview/ProjectOverview'
import ModelRunConfig from 'containers/Configuration/Forms/ModelRunConfig'
import SosModelConfig from 'containers/Configuration/Forms/SosModelConfig'
import SectorModelConfig from 'containers/Configuration/Forms/SectorModelConfig'
import ScenarioConfig from 'containers/Configuration/Forms/ScenarioConfig'

import store from 'store/store.js'
import history from './history'

import 'bootstrap/dist/css/bootstrap.min.css'
import '../static/css/main.css'
import 'rc-slider/assets/index.css'
import 'react-table/react-table.css'

render(
    <Provider store={store}>
        <Router history={history}>
            <div className="container-fluid">
                <div className="row">
                    <Route path="/" component={Nav}/>
                    <main role="main" className="col-12 col-md-9 col-xl-8 py-3 px-4">
                        <Switch>
                            <Route exact path="/" component={Welcome}/>
                            <Route exact path="/jobs/:param?" component={JobsOverview}/>
                            <Route exact strict path="/jobs/runner/:name" component={JobRunner}/>
                            <Route exact strict path="/configure/:name" component={ProjectOverview}/>
                            <Route exact strict path="/configure/model-runs/:name" component={ModelRunConfig}/>
                            <Route exact strict path="/configure/sos-models/:name" component={SosModelConfig}/>
                            <Route exact strict path="/configure/sector-models/:name" component={SectorModelConfig}/>
                            <Route exact strict path="/configure/scenarios/:name" component={ScenarioConfig}/>
                            <Route component={NotFound} />
                        </Switch>
                    </main>
                </div>
            </div>
        </Router>
    </Provider>,
    document.getElementById('root')
)
