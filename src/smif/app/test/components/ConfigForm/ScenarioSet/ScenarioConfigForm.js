import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow, render } from 'enzyme'

import ScenarioConfigForm from '../../../../src/components/ConfigForm/ScenarioSet/ScenarioConfigForm.js'

import {scenario, scenario_set} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

describe('<ScenarioConfigForm />', () => {

    const correctRender = shallow(<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} />)
    const dataMissingRender = shallow(<ScenarioConfigForm scenario={empty_object} scenarioSet={empty_object} />)

    it('renders scenario.name', () => {
        const scenario_name = correctRender.find('[id="scenario_name"]')
        expect(scenario_name.html()).to.contain(scenario.name)
    })

    it('renders scenario.description', () => {
        const scenario_description = correctRender.find('[id="scenario_description"]')
        expect(scenario_description.html()).to.contain(scenario.description)
    })

    it('loads properties ', () => {
        const wrapper = mount(<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} />)
        expect(wrapper.props()['scenario']).to.equal(scenario)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const wrapper = mount(<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} saveScenario={onSaveClick} />)

        wrapper.find('input#saveScenario').simulate('click')
        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.equal(scenario)
    })

    it('save callback with changed properties', () => {
        const changed_scenario = {
            name: 'new_scenario_name',
            description: 'new_description',
            filename: 'new_filename',
            facets: scenario['facets'],
            scenario_set: 'new_scenario_set'
        }

        const onSaveClick = sinon.spy()
        const wrapper = mount(<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} saveScenario={onSaveClick} />)

        wrapper.find('#scenario_name').simulate('change', { target: { name: 'name', value: changed_scenario['name'] } })
        wrapper.find('#scenario_name').simulate('change', { target: { name: 'description', value: changed_scenario['description'] } })
        wrapper.find('#scenario_name').simulate('change', { target: { name: 'filename', value: changed_scenario['filename'] } })
        wrapper.find('#scenario_name').simulate('change', { target: { name: 'scenario_set', value: changed_scenario['scenario_set'] } })
        wrapper.find('input#saveScenario').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_scenario)
    })

    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount(<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} cancelScenario={onCancelClick} />)

        const nodes = wrapper.find('input#cancelScenario')
        // Need to specify element is input (not just by id)
        // and only find the actual button
        // calling hostNodes would also filter out React Components
        // e.g. nodes = wrapper.find('#cancelScenario').hostNodes()
        // To check the rendered output use:
        // console.log(nodes.debug())
        nodes.simulate('click')
        expect(onCancelClick.calledOnce).to.equal(true);
    })

    it('unmount', () => {
        var wrapper = mount((<ScenarioConfigForm scenario={scenario} scenarioSet={scenario_set} />))

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
