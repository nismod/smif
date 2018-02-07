import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'
import NarrativeSelector from '../../../../src/components/ConfigForm/SosModelRun/NarrativeSelector.js'

import {sos_model_run, sos_models, narratives, sos_model} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var render, warning

describe('<NarrativeSelector />', () => {

    it('renders narrative sets and narrative name', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={narratives} />)

        expect(render.html()).to.contain(narratives[0].name)
        expect(render.html()).to.contain(narratives[0].narrative_set)
    })

    it('correctly displays the sos_model_run configuration', () => {
        const wrapper = mount(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={narratives} />)

        for (var i = 0; i < narratives.length; i++) {
            if (sos_model_run['narratives'][narratives[i]['narrative_set']].includes(narratives[i]['name'])) {
                expect(wrapper.find('[id="' + narratives[i]['name'] + '"]').props().defaultChecked).to.equal(true)
            } else {
                expect(wrapper.find('[id="' + narratives[i]['name'] + '"]').props().defaultChecked).to.equal(false)
            }
        }
    })

    it('onChange callback', () => {
        const onChange = sinon.spy()
        const wrapper = mount(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={narratives} onChange={onChange} />)

        for (var i = 0; i < narratives.length; i++) {
            if (sos_model_run['narratives'][narratives[i]['narrative_set']].includes(narratives[i]['name'])) {
                wrapper.find('[id="' + narratives[i]['name'] + '"]').simulate('click', { target: { name: narratives[i]['narrative_set'], checked: false }})
                expect(onChange.args[i][0]).to.equal(narratives[i]['narrative_set'])
                expect(onChange.args[i][1]).to.equal(narratives[i]['name'])
                expect(onChange.args[i][2]).to.equal(false)
            } else {
                wrapper.find('[id="' + narratives[i]['name'] + '"]').simulate('click', { target: { name: narratives[i]['narrative_set'], checked: true }})
                expect(onChange.args[i][0]).to.equal(narratives[i]['narrative_set'])
                expect(onChange.args[i][1]).to.equal(narratives[i]['name'])
                expect(onChange.args[i][2]).to.equal(true)
            }
        }
    })

    it('warning no sosModel selected', () => {
        var custom_sos_model_run = Object.assign({}, sos_model_run)
        custom_sos_model_run.sos_model = ''

        render = shallow(<NarrativeSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModel configured in the SosModelRun')

        custom_sos_model_run.sos_model = null

        render = shallow(<NarrativeSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModel configured in the SosModelRun')
    })

    it('warning no narrativeSets in sosModel', () => {
        var custom_sos_models = sos_models.map(a => Object.assign({}, a)) // deep copy variant of -> var custom_sos_models = sos_models.slice(0)
        custom_sos_models[0].narrative_sets = []

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-info"]')

        expect(warning.html()).to.contain('There are no NarrativeSets configured in the SosModel')

        custom_sos_models[0].narrative_sets = null

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-info"]')

        expect(warning.html()).to.contain('There are no NarrativeSets configured in the SosModel')
    })

    it('warning no sosModelRun configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={empty_object} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')

        render = shallow(<NarrativeSelector sosModelRun={null} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')
    })

    it('warning no sosModels configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={empty_array} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={null} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')
    })

    it('warning no narratives configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={empty_array} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no Narratives configured')

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={null} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no Narratives configured')
    })
})
