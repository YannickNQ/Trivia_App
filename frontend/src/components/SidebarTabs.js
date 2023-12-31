import React from 'react';
import { Tab } from 'semantic-ui-react';
import TabStyles from './styles/TabStyles';
import DropdownSelectionStyles from './styles/DropdownSelectionStyles';
import MapViewDropdown from './MapViewDropdown';
import RegionSearch from './RegionSearch';
import QuizBox from './quizBox/QuizBox';
import ChoroplethToggles from './ChoroplethToggles';
// import About from './About';

const panes = [
  {
    menuItem: { key: 'Quiz', content: 'Quiz' },
    render: () => (
      <Tab.Pane attached={false}>
        <DropdownSelectionStyles>
          <MapViewDropdown />
          <RegionSearch />
        </DropdownSelectionStyles>
        <QuizBox />
      </Tab.Pane>
    ),
  }
];

const SidebarTabs = () => (
  <TabStyles menu={{ secondary: true, pointing: true }} panes={panes} />
);

export default SidebarTabs;
