import React from 'react'
import {expect} from 'chai'
import {shallow, mount} from 'enzyme'
import {describe, it} from 'mocha'

import ReactModal from 'react-modal'
import SosModelConfigForm from '../../../src/components/ConfigForm/SosModelConfigForm.js'

import {sos_model, sector_models, scenario_sets, narrative_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe.skip('<SosModelConfigForm />', () => {

    it('renders', () => {
        const correctRender = shallow(<SosModelConfigForm sosModel={sos_model} sectorModels={sector_models} scenarioSets={scenario_sets} narrativeSets={narrative_sets} />)
        const dataMissingRender = shallow(<SosModelConfigForm sosModel={empty_object} sectorModels={empty_array} scenarioSets={empty_array} narrativeSets={empty_array} />)

        // renders sos_model.name
        let sos_model_name = correctRender.find('[id="sos_model_name"]')
        expect(sos_model_name.html()).to.contain(sos_model.name)

        // renders sos_model.name when data missing
        sos_model_name = dataMissingRender.find('[id="sos_model_name"]')
        expect(sos_model_name.html()).to.contain('id="sos_model_name"')

        // renders sos_model.description
        let sos_model_description = correctRender.find('[id="sos_model_description"]')
        expect(sos_model_description.html()).to.contain(sos_model.description)

        // renders sos_model.description when data missing
        sos_model_description = dataMissingRender.find('[id="sos_model_description"]')
        expect(sos_model_description.html()).to.contain('id="sos_model_description"')
    })

    it('add / remove dependency', () => {
        sos_model.dependencies = []
        const wrapper = mount(
            <SosModelConfigForm 
                sosModel={sos_model} 
                sectorModels={sector_models} 
                scenarioSets={scenario_sets} 
                narrativeSets={narrative_sets} 
            />)

        // Open popup
        wrapper.find('input#btn_add_dependency').simulate('click')
        let popup_add_dependency = wrapper.find(ReactModal).find('[id="popup_add_dependency"]')
        expect(popup_add_dependency.exists()).to.equal(true)

        // Select dependency
        wrapper.find('select#select_source').simulate('change', { target: { name: 'SourceModel', value: 'population'} })
        wrapper.find('select#select_source_output').simulate('change', { target: { name: 'SourceOutput', value: 'population'} })
        wrapper.find('select#select_sink').simulate('change', { target: { name: 'SinkModel', value: 'water_supply'} })
        wrapper.find('select#select_sink_input').simulate('change', { target: { name: 'SinkInput', value: 'population'} })
        
        wrapper.find('input#btn_save_dependency').simulate('submit')
        
        // Check if dependency was added
        expect(wrapper.state().selectedSosModel.dependencies[0]).to.deep.equal({
            source_model: 'population',
            source_model_output: 'population',
            sink_model: 'water_supply',
            sink_model_input: 'population'
        })

        // Delete dependency
        wrapper.find('button#btn_del_0').simulate('click')
        wrapper.find('input#deleteButton').simulate('click')
        expect(wrapper.state().selectedSosModel.dependencies).to.deep.equal([])    
    })
})
