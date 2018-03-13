import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'
import SosModelRunConfigForm from '../../../src/components/ConfigForm/SosModelRunConfigForm.js'

import {sos_model_run, sos_models, scenarios, narratives} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<SosModelRunConfigForm />', () => {

    const correctRender = shallow(<SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} />)
    const dataMissingRender = shallow(<SosModelRunConfigForm sosModelRun={empty_object} sosModels={empty_array} scenarios={empty_array} narratives={empty_array} />)

    it('renders sos_model_run.name', () => {
        const sos_model_run_name = correctRender.find('[id="sos_model_run_name"]')
        expect(sos_model_run_name.html()).to.contain(sos_model_run.name)
    })

    it('renders sos_model_run.name when data missing', () => {
        const sos_model_run_name = dataMissingRender.find('[id="sos_model_run_name"]')
        expect(sos_model_run_name.html()).to.contain(`id="sos_model_run_name"`)
    })

    it('renders sos_model_run.description', () => {
        const sos_model_run_description = correctRender.find('[id="sos_model_run_description"]')
        expect(sos_model_run_description.html()).to.contain(sos_model_run.description)
    })

    it('renders sos_model_run.description when data missing', () => {
        const sos_model_run_description = dataMissingRender.find('[id="sos_model_run_description"]')
        expect(sos_model_run_description.html()).to.contain(`id="sos_model_run_description"`)
    })

    it('loads properties ', () => {
        const wrapper = mount(<SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} />)
        expect(wrapper.props()['sosModelRun']).to.equal(sos_model_run)
    })

    it('scenarios correctly managed', () => {
        // This test assumes that the population scenario is available with Central Population (Medium) already configured
        const wrapper = mount(<SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} />)

        // Check excisting scenario to be loaded
        expect(wrapper.find('[id="radio_population_Central Population (Low)"]').props().checked).to.equal(false)
        expect(wrapper.find('[id="radio_population_Central Population (Medium)"]').props().checked).to.equal(true)
        expect(wrapper.find('[id="radio_population_Central Population (High)"]').props().checked).to.equal(false)
        expect(wrapper.state()['selectedSosModelRun']['scenarios']['population']).to.equal("Central Population (Medium)")

        // Change scenario
        wrapper.find('[id="radio_population_Central Population (High)"]').simulate('click', { target: { name: 'population', value: "Central Population (High)"}})
        expect(wrapper.state()['selectedSosModelRun']['scenarios']['population']).to.equal("Central Population (High)")
    })

    it('narratives correctly managed', () => {
        // This test assumes that the technology narrative is available with High Tech Demand Side Management already configured
        const wrapper = mount(<SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} />)

        // Check excisting narrative to be loaded
        expect(wrapper.find('[id="High Tech Demand Side Management"]').props().defaultChecked).to.equal(true)
        expect(wrapper.find('[id="Low Tech Demand Side Management"]').props().defaultChecked).to.equal(false)

        // Remove the one existing narrative
        wrapper.find('[id="High Tech Demand Side Management"]').simulate('click', { target: { name: 'technology', checked: false }})
        expect(wrapper.state()['selectedSosModelRun']['narratives']).to.deep.equal({})

        // Select two narratives
        wrapper.find('[id="High Tech Demand Side Management"]').simulate('click', { target: { name: 'technology', checked: true }})
        wrapper.find('[id="Low Tech Demand Side Management"]').simulate('click', { target: { name: 'technology', checked: true }})
        expect(wrapper.state()['selectedSosModelRun']['narratives']).to.deep.equal({technology: ['High Tech Demand Side Management', 'Low Tech Demand Side Management']})
    })

})
