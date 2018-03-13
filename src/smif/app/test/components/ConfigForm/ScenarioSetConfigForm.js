import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'

import ScenarioSetConfigForm from '../../../src/components/ConfigForm/ScenarioSetConfigForm.js'

import {scenario_set, scenarios} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<ScenarioSetConfigForm />', () => {

    const correctRender = shallow(<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} />)
    const dataMissingRender = shallow(<ScenarioSetConfigForm scenarioSet={empty_object} scenarios={empty_array}/>)

    it('renders scenario_set.name', () => {
        const scenario_set_name = correctRender.find('[id="scenario_set_name"]')
        expect(scenario_set_name.html()).to.contain(scenario_set.name)
    })

    it('renders scenario_set.description', () => {
        const scenario_set_description = correctRender.find('[id="scenario_set_description"]')
        expect(scenario_set_description.html()).to.contain(scenario_set.description)
    })

    it('loads properties ', () => {
        const wrapper = mount((<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} />))
        expect(wrapper.props()['scenarioSet']).to.equal(scenario_set)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const onDeleteScenario = sinon.spy()
        const onCreateScenario = sinon.spy()
        const wrapper = mount((<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} saveScenarioSet={onSaveClick} deleteScenario={onDeleteScenario} createScenario={onCreateScenario} />))
    
        wrapper.find('[id="saveButton"]').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.equal(scenario_set)
    })

    it('save callback with changed properties', () => {
        const changed_scenario_set = {
            name: 'new_name',
            description: 'new_description',
            facets: [
                {
                    description: 'Central Population for the UK',
                    name: 'population'
                }
            ]
        }

        const onSaveClick = sinon.spy()
        const onDeleteScenario = sinon.spy()
        const onCreateScenario = sinon.spy()
        const wrapper = mount((<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} saveScenarioSet={onSaveClick} deleteScenario={onDeleteScenario} createScenario={onCreateScenario}/>))

        wrapper.find('[id="scenario_set_name"]').simulate('change', { target: { name: 'name', value: changed_scenario_set['name'] } })
        wrapper.find('[id="scenario_set_description"]').simulate('change', { target: { name: 'description', value: changed_scenario_set['description'] } })
        wrapper.find('[id="saveButton"]').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_scenario_set)
    })
    
    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount((<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} cancelScenarioSet={onCancelClick} />))
    
        wrapper.find('[id="cancelButton"]').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount((<ScenarioSetConfigForm scenarioSet={scenario_set} scenarios={scenarios} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
