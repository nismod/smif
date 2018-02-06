import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'

import ScenarioConfigForm from '../../../src/components/ConfigForm/ScenarioConfigForm.js'

import {scenario, scenario_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<ScenarioConfigForm />', () => {

    const correctRender = shallow(<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} />)
    const dataMissingRender = shallow(<ScenarioConfigForm scenario={empty_object} scenarioSets={empty_array} />)

    it('renders scenario.name', () => {
        const scenario_name = correctRender.find('[id="scenario_name"]')
        expect(scenario_name.html()).to.contain(scenario.name)
    })

    it('renders scenario.name when data missing', () => {
        const scenario_name = dataMissingRender.find('[id="scenario_name"]')
        expect(scenario_name.html()).to.contain(`id="scenario_name"`)
    })

    it('renders scenario.description', () => {
        const scenario_description = correctRender.find('[id="scenario_description"]')
        expect(scenario_description.html()).to.contain(scenario.description)
    })

    it('renders scenario.description when data missing', () => {
        const scenario_description = dataMissingRender.find('[id="scenario_description"]')
        expect(scenario_description.html()).to.contain(`id="scenario_description"`)
    })

    it('loads properties ', () => {
        const wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} />))
        expect(wrapper.props()['scenario']).to.equal(scenario)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} saveScenario={onSaveClick} />))
    
        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.equal(scenario)
    })

    it('save callback with changed properties', () => {
        const changed_scenario = {
            name: 'new_scenario_name',
            description: 'new_description',
            filename: 'new_filename',
            parameters: scenario['parameters'],
            scenario_set: 'new_scenario_set'
        }

        const onSaveClick = sinon.spy()
        const wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} saveScenario={onSaveClick} />))

        wrapper.find('[id="scenario_name"]').simulate('change', { target: { name: 'name', value: changed_scenario['name'] } })
        wrapper.find('[id="scenario_name"]').simulate('change', { target: { name: 'description', value: changed_scenario['description'] } })
        wrapper.find('[id="scenario_name"]').simulate('change', { target: { name: 'filename', value: changed_scenario['filename'] } })
        wrapper.find('[id="scenario_name"]').simulate('change', { target: { name: 'scenario_set', value: changed_scenario['scenario_set'] } })
        wrapper.find('[id="saveButton"]').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_scenario)
    })
    
    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} cancelScenario={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
