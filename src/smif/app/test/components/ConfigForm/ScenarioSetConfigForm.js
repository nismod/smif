import React from 'react'
import sinon from 'sinon'
import {expect} from 'chai'
import {mount} from 'enzyme'
import {describe, it} from 'mocha'

import ReactModal from 'react-modal'
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

    it('add / edit / remove Facet', () => {
        let wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={empty_array} />)

        // Check if facet is not there
        expect(wrapper.find('tr#facets_property_1').exists()).to.equal(false)

        // Open the Add facets input popup
        wrapper.find('input#btn_add_facet').simulate('click')

        // Check if the form opens
        let popup_add_facet = wrapper.find('[id="popup_add_facet"]')
        expect(popup_add_facet.exists()).to.equal(true)

        // Fill in form
        wrapper.find('input#facet_name').simulate('change', { target: { name: 'name', value: 'test_name'} })
        wrapper.find('textarea#facet_description').simulate('change', { target: { name: 'description', value: 'test_description'} })

        // Submit form
        wrapper.find('input#btn_facet_save').simulate('click')

        // Check if facet was added
        expect(wrapper.state().scenarioSet.facets[1].name).to.equal('test_name')
        expect(wrapper.state().scenarioSet.facets[1].description).to.equal('test_description')
        
        // Check if facet appears in list
        expect(wrapper.find('tr#facets_property_1').exists()).to.equal(true)
        expect(wrapper.find('tr#facets_property_1').html()).to.include('test_name')
        expect(wrapper.find('tr#facets_property_1').html()).to.include('test_description')

        // Edit facet
        wrapper.find('button#btn_edit_test_name').simulate('click')
        wrapper.find('textarea#facet_description').simulate('change', { target: { name: 'description', value: 'edited_test_description'} })
        wrapper.find('input#btn_facet_save').simulate('click')
        expect(wrapper.find('tr#facets_property_1').html()).to.include('edited_test_description')

        // Remove the facet
        wrapper.find('tr#facets_property_1').find('button#btn_del_test_name').simulate('click')
        let popup_delete = wrapper.find('[id="popup_delete"]')
        expect(popup_delete.exists()).to.equal(true)
        wrapper.find('input#deleteButton').simulate('click')

        // Check if facet was removed
        expect(wrapper.find('tr#facets_property_1').exists()).to.equal(false)
    })

    it('add / edit / remove Scenario', () => {
        const createScenario = sinon.spy()
        const deleteScenario = sinon.spy()
        const saveScenario = sinon.spy()

        let wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={empty_array}
            createScenario={createScenario}
            deleteScenario={deleteScenario}
            saveScenario={saveScenario} />)

        // Check if facet is not there
        expect(wrapper.find('tr#facets_property_1').exists()).to.equal(false)

        // Open the Add facets input popup
        wrapper.find('input#btn_createScenario').simulate('click')

        // Check if the form opens
        let popup_scenario_config = wrapper.find('[id="popup_scenario_config_form"]')
        expect(popup_scenario_config.exists()).to.equal(true)

        // Fill in form
        let my_scenario = {
            name: 'test_scenario_name',
            description: 'test_scenario_description',
            facets: [
                {
                    name: 'test_facet_name',
                    filename: 'test_filename',
                    spatial_resolution: 'test_spatial_res',
                    temporal_resolution: 'test_temporal_res',
                    units: 'test_units'
                }
            ],
            scenario_set: 'population'
        }

        wrapper.find('input#scenario_name').simulate('change', { target: { name: 'name', value: my_scenario.name } })
        wrapper.find('textarea#scenario_description').simulate('change', { target: { name: 'description', value: my_scenario.description } })
        wrapper.find('input#scenario_facet_name').simulate('change', { target: { name: 'facet_name', value: my_scenario.facets[0].name } })
        wrapper.find('input#scenario_filename').simulate('change', { target: { name: 'facet_filename', value: my_scenario.facets[0].filename } })
        wrapper.find('input#scenario_units').simulate('change', { target: { name: 'facet_units', value: my_scenario.facets[0].units}})
        wrapper.find('input#scenario_spatial_res').simulate('change', { target: { name: 'facet_spatial_resolution', value: my_scenario.facets[0].spatial_resolution} })
        wrapper.find('input#scenario_temp_res').simulate('change', { target: { name: 'facet_temporal_resolution', value: my_scenario.facets[0].temporal_resolution} })

        // Submit form
        wrapper.find('input#btn_save_scenario').simulate('click')

        // Check if createScenario callback was called
        expect(createScenario.args[0][0]).to.deep.equal(my_scenario)

        // Pretend that scenario was added by callback
        wrapper = mount(<ScenarioSetConfigForm
            sosModelRuns={empty_array}
            sosModels={empty_array}
            scenarioSet={scenario_set}
            scenarios={[my_scenario]}
            createScenario={createScenario}
            deleteScenario={deleteScenario}
            saveScenario={saveScenario} />)

        // Check if facet appears in list
        expect(wrapper.find('tr#scenarios_property_0').exists()).to.equal(true)
        expect(wrapper.find('tr#scenarios_property_0').html()).to.include(my_scenario.name)
        expect(wrapper.find('tr#scenarios_property_0').html()).to.include(my_scenario.description)

        // Edit the scenario
        wrapper.find('button#btn_edit_test_scenario_name').simulate('click')
        wrapper.find('textarea#scenario_description').simulate('change', { target: { name: 'description', value: 'edited_scenario_test_description'} })
        wrapper.find('input#btn_save_scenario').simulate('click')
        my_scenario.description = 'edited_scenario_test_description'
        expect(saveScenario.args[0][0]).to.deep.equal(my_scenario)

        // Remove the Scenario
        wrapper.find('tr#scenarios_property_0').find('button#btn_del_' + my_scenario.name).simulate('click')
        let popup_delete = wrapper.find('[id="popup_delete"]')
        expect(popup_delete.exists()).to.equal(true)
        wrapper.find('input#deleteButton').simulate('click')

        // Check if Scenario delete callback was invoked
        expect(deleteScenario.args[0][0]).to.equal(my_scenario.name)
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
