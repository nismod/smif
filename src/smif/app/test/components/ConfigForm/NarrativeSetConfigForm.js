import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import NarrativeSetConfigForm from '../../../src/components/ConfigForm/NarrativeSetConfigForm.js'

import {narrative_set} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<NarrativeSetConfigForm />', () => {

    const correctRender = shallow(<NarrativeSetConfigForm narrativeSet={narrative_set} />)
    const dataMissingRender = shallow(<NarrativeSetConfigForm narrativeSet={empty_object} />)

    it('renders narrative_set.name', () => {
        const narrative_set_name = correctRender.find('[id="narrative_set_name"]')
        expect(narrative_set_name.html()).to.contain(narrative_set.name)
    })

    it('renders narrative_set.name when data missing', () => {
        const narrative_set_name = dataMissingRender.find('[id="narrative_set_name"]')
        expect(narrative_set_name.html()).to.contain(`id="narrative_set_name"`)
    })

    it('renders narrative_set.description', () => {
        const narrative_set_description = correctRender.find('[id="narrative_set_description"]')
        expect(narrative_set_description.html()).to.contain(narrative_set.description)
    })

    it('renders narrative_set.description when data missing', () => {
        const narrative_set_description = dataMissingRender.find('[id="narrative_set_description"]')
        expect(narrative_set_description.html()).to.contain(`id="narrative_set_description"`)
    })
})
