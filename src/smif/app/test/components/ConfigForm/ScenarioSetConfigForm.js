import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount, ReactWrapper} from 'enzyme'
import {describe, it} from 'mocha'

import ReactModal from 'react-modal';
import ScenarioSetConfigForm from '../../../src/components/ConfigForm/ScenarioSetConfigForm.js'

import {scenario_set, scenarios} from '../../helpers.js'
import {empty_array} from '../../helpers.js'

describe('<ScenarioSetConfigForm />', () => {

    it('renders scenario_set and scenarios', () => {
        let wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios} />)

        const scenario_set_name = wrapper.find('[id="scenario_set_name"]')
        expect(scenario_set_name.html()).to.contain(scenario_set.name)

        const scenario_set_description = wrapper.find('[id="scenario_set_description"]')
        expect(scenario_set_description.html()).to.contain(scenario_set.description)

        const scenario_set_facets_0 = wrapper.find('[id="facets_property_0"]')
        expect(scenario_set_facets_0.html()).to.contain(scenario_set.facets[0].name)
        expect(scenario_set_facets_0.html()).to.contain(scenario_set.facets[0].description)

        const scenarios_0 = wrapper.find('[id="scenarios_property_0"]')
        expect(scenarios_0.html()).to.contain(scenarios[0].name)
        expect(scenarios_0.html()).to.contain(scenarios[0].description)
    })

    it('renders with missing props', () => {
        let wrapper = mount(<ScenarioSetConfigForm/>)
        expect(wrapper.html()).to.contain('This Scenario Set does not exist.')
    })

    it('renders scenario_set with missing facets property (auto-fix)', () => {
        let my_scenarioSet = Object.assign({}, scenario_set)
        delete my_scenarioSet.facets

        let wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={my_scenarioSet}
            scenarios={scenarios} />)

        expect(wrapper.html()).to.contain('There are no facets configured')
    })

    it('create Scenario', () => {
        let wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios} />)

        // Check if popup is not open
        expect(wrapper.state('editScenarioPopupIsOpen')).to.equal(false)
        let popup_scenario = wrapper.find(ReactModal).find('[id="popup_scenario_config_form"]')
        expect(popup_scenario.exists()).to.equal(false)

        // Open the popup
        wrapper.find('input#btn_createScenario').simulate('click')

        // Check if popup was opened
        expect(wrapper.state('editScenarioPopupIsOpen')).to.equal(true)
        popup_scenario = wrapper.find(ReactModal).find('[id="popup_scenario_config_form"]')
        expect(popup_scenario.exists()).to.equal(true)
    })

    it('save callback on saveButton click', () => {
        const onSaveClick = sinon.spy()
        const onDeleteScenario = sinon.spy()
        const onCreateScenario = sinon.spy()
        const wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios}
            saveScenarioSet={onSaveClick}
            deleteScenario={onDeleteScenario}
            createScenario={onCreateScenario} />)

        wrapper.find('input#btn_saveScenarioSet').simulate('click')
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
        const wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios}
            saveScenarioSet={onSaveClick}
            deleteScenario={onDeleteScenario}
            createScenario={onCreateScenario}/>)

        wrapper.find('[id="scenario_set_name"]').simulate('change', { target: { name: 'name', value: changed_scenario_set['name'] } })
        wrapper.find('[id="scenario_set_description"]').simulate('change', { target: { name: 'description', value: changed_scenario_set['description'] } })
        wrapper.find('input#btn_saveScenarioSet').simulate('click')

        expect(onSaveClick).to.have.property('callCount', 1)
        expect(onSaveClick.args[0][0]).to.deep.equal(changed_scenario_set)
    })

    it('cancel callback on cancelButton click', () => {
        const onCancelClick = sinon.spy()
        const wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios}
            cancelScenarioSet={onCancelClick} />)

        wrapper.find('input#btn_cancelScenarioSet').simulate('click')
        expect(onCancelClick).to.have.property('callCount', 1)
    })

    it('unmount', () => {
        var wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={scenarios} />)

        wrapper = wrapper.unmount()
        expect(wrapper.html()).to.be.null
    })
})
