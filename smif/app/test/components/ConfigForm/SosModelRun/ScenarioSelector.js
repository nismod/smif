import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import ScenarioSelector from '../../../../src/components/ConfigForm/SosModelRun/ScenarioSelector.js'

import {sos_model_run, sos_models, scenarios} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var render, warning

describe('<ScenarioSelector />', () => {

    it('warning no sosModel selected', () => {
        var custom_sos_model_run = Object.assign({}, sos_model_run)
        custom_sos_model_run.sos_model = ""

        render = shallow(<ScenarioSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There is no SosModel configured in the SosModelRun")

        custom_sos_model_run.sos_model = null

        render = shallow(<ScenarioSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There is no SosModel configured in the SosModelRun")
    })

    it('warning no scenariosets in sosModel', () => {
        var custom_sos_models = sos_models.slice(0)
        custom_sos_models[0].scenario_sets = []

        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no ScenarioSets configured in the SosModel")

        custom_sos_models[0].scenario_sets = null

        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no ScenarioSets configured in the SosModel")
    })

    it('warning no sosModelRun configured', () => {
        render = shallow(<ScenarioSelector sosModelRun={empty_object} sosModels={sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There is no SosModelRun configured")

        render = shallow(<ScenarioSelector sosModelRun={null} sosModels={sos_models} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There is no SosModelRun configured")
    })

    it('warning no sosModels configured', () => {
        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={empty_array} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no SosModels configured")

        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={null} scenarios={scenarios} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no SosModels configured")
    })

    it('warning no scenarios configured', () => {
        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={sos_models} scenarios={empty_array} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no Scenarios configured")

        render = shallow(<ScenarioSelector sosModelRun={sos_model_run} sosModels={sos_models} scenarios={null} />)
        warning = render.find('[id="scenario_selector_warning"]')

        expect(warning.html()).to.contain("There are no Scenarios configured")
    })
})