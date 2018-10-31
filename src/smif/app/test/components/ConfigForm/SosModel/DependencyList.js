import React from 'react'
import {expect} from 'chai'
import {shallow, mount} from 'enzyme'
import {describe, it} from 'mocha'

import ReactModal from 'react-modal'
import DependencyList from '../../../../src/components/ConfigForm/SosModel/DependencyList.js'

import {sector_models, scenario_sets, sos_model} from '../../../helpers.js'

var wrapper

describe.skip('<DependencyList />', () => {

    it('warning no sectorModel configured', () => {
        wrapper = shallow(<DependencyList sectorModels={null} scenarioSets={scenario_sets} dependencies={sos_model.dependencies} selectedSectorModels={sos_model.sector_models} selectedScenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('sectorModels are undefined')
        
        wrapper = shallow(<DependencyList sectorModels={sector_models} scenarioSets={null} dependencies={sos_model.dependencies} selectedSectorModels={sos_model.sector_models} selectedScenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('scenarioSets are undefined')

        wrapper = shallow(<DependencyList sectorModels={sector_models} scenarioSets={scenario_sets} dependencies={null} selectedSectorModels={sos_model.sector_models} selectedScenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('Dependencies are undefined')

        wrapper = shallow(<DependencyList sectorModels={sector_models} scenarioSets={scenario_sets} dependencies={sos_model.dependencies} selectedSectorModels={null} selectedScenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('selectedSectorModels are undefined')

        wrapper = shallow(<DependencyList sectorModels={sector_models} scenarioSets={scenario_sets} dependencies={sos_model.dependencies} selectedSectorModels={sos_model.sector_models} selectedScenarioSets={null}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('selectedScenarioSets are undefined')
    })
    
    it('Renders correct', () => {
        wrapper = mount(<DependencyList 
            sectorModels={sector_models} 
            scenarioSets={scenario_sets} 
            dependencies={sos_model.dependencies} 
            selectedSectorModels={sos_model.sector_models} 
            selectedScenarioSets={sos_model.scenario_sets}
        />)
        
        // Open the add dependency popup
        wrapper.find('input#btn_add_dependency').simulate('click')
        
        // Check if popup was opened
        expect(wrapper.state('CreateDependencypopupIsOpen')).to.equal(true)
        let popup_add_dependency = wrapper.find(ReactModal).find('[id="popup_add_dependency"]')
        expect(popup_add_dependency.exists()).to.equal(true)

        // Check if the right options are there
        expect(wrapper.find('select#select_source').html()).to.equal('' +
            '<select id="select_source" class="form-control" name="SourceModel">' + 
                '<option disabled="" value="none">Please select a source</option>' + 
                '<option disabled="">Sector Model</option>' + 
                '<option value="' + sector_models[0].name + '">' + sector_models[0].name + '</option>' + 
                '<option value="' + sector_models[1].name + '">' + sector_models[1].name + '</option>' + 
                '<option disabled="">Scenario Set</option>' + 
                '<option value="' + sos_model.scenario_sets[0] + '">' + sos_model.scenario_sets[0] + '</option>' + 
                '<option value="' + sos_model.scenario_sets[1] + '">' + sos_model.scenario_sets[1] + '</option>' + 
            '</select>')

        expect(wrapper.find('select#select_source_output').html()).to.equal('' +
            '<select id="select_source_output" class="form-control" name="SourceOutput">' + 
                '<option disabled="" value="none">None</option>' + 
            '</select>')

        expect(wrapper.find('select#select_sink').html()).to.equal('' +
            '<select id="select_sink" class="form-control" name="SinkModel">' + 
                '<option disabled="" value="none">Please select a sink</option>' + 
                '<option disabled="">Sector Model</option>' + 
                '<option value="' + sector_models[0].name + '">' + sector_models[0].name + '</option>' + 
                '<option value="' + sector_models[1].name + '">' + sector_models[1].name + '</option>' + 
            '</select>')

        expect(wrapper.find('select#select_sink_input').html()).to.equal('' +
            '<select id="select_sink_input" class="form-control" name="SinkInput">' + 
                '<option disabled="" value="none">None</option>' + 
            '</select>')
    })

    it('Check dynamic selection rendering', () => {
        wrapper = mount(<DependencyList 
            sectorModels={sector_models} 
            scenarioSets={scenario_sets} 
            dependencies={sos_model.dependencies} 
            selectedSectorModels={sos_model.sector_models} 
            selectedScenarioSets={sos_model.scenario_sets}
        />)
        
        // Open the add dependency popup
        wrapper.find('input#btn_add_dependency').simulate('click')
        
        // Check if popup was opened
        expect(wrapper.state('CreateDependencypopupIsOpen')).to.equal(true)
        let popup_add_dependency = wrapper.find(ReactModal).find('[id="popup_add_dependency"]')
        expect(popup_add_dependency.exists()).to.equal(true)

        // Select the first Source Model
        wrapper.find('select#select_source').simulate('change', { target: { name: 'SourceModel', value: sector_models[0].name } })

        // When the Source Model is selected, its outputs should be listed
        expect(wrapper.find('select#select_source_output').html()).to.equal('' +
            '<select id="select_source_output" class="form-control" name="SourceOutput">' + 
                '<option disabled="" value="none">Please select a source output</option>' + 
                '<option value="' + sector_models[0].outputs[0].name + '">' + sector_models[0].outputs[0].name + '</option>' + 
                '<option value="' + sector_models[0].outputs[1].name + '">' + sector_models[0].outputs[1].name + '</option>' + 
                '<option value="' + sector_models[0].outputs[2].name + '">' + sector_models[0].outputs[2].name + '</option>' + 
            '</select>')

        // And the model should be removed from the Sink Inputs (to avoid circular dependencies)
        expect(wrapper.find('select#select_sink').html()).to.equal('' +
            '<select id="select_sink" class="form-control" name="SinkModel">' + 
                '<option disabled="" value="none">Please select a sink</option>' + 
                '<option disabled="">Sector Model</option>' + 
                '<option value="' + sector_models[1].name + '">' + sector_models[1].name + '</option>' + 
            '</select>')

        // Select the second Source Model
        wrapper.find('select#select_source').simulate('change', { target: { name: 'SourceModel', value: sector_models[1].name } })

        // Outputs should be updated
        expect(wrapper.find('select#select_source_output').html()).to.equal('' +
        '<select id="select_source_output" class="form-control" name="SourceOutput">' + 
            '<option disabled="" value="none">Please select a source output</option>' + 
            '<option value="' + sector_models[1].outputs[0].name + '">' + sector_models[1].outputs[0].name + '</option>' + 
            '<option value="' + sector_models[1].outputs[1].name + '">' + sector_models[1].outputs[1].name + '</option>' + 
        '</select>')

        // And the model should be removed from the Sink Inputs (to avoid circular dependencies)
        expect(wrapper.find('select#select_sink').html()).to.equal('' +
        '<select id="select_sink" class="form-control" name="SinkModel">' + 
            '<option disabled="" value="none">Please select a sink</option>' + 
            '<option disabled="">Sector Model</option>' + 
            '<option value="' + sector_models[0].name + '">' + sector_models[0].name + '</option>' + 
        '</select>')

        // Select a scenario set
        wrapper.find('select#select_source').simulate('change', { target: { name: 'SourceModel', value: sos_model.scenario_sets[0] } })
    
        // Outputs should be updated
        expect(wrapper.find('select#select_source_output').html()).to.equal('' +
        '<select id="select_source_output" class="form-control" name="SourceOutput">' + 
            '<option disabled="" value="none">Please select a source output</option>' + 
            '<option value="' + scenario_sets[1].facets[0].name + '">' + scenario_sets[1].facets[0].name + '</option>' + 
        '</select>')
    })
})
