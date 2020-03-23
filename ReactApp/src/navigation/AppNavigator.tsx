import {createAppContainer, createSwitchNavigator} from "react-navigation";
import {ItemScreens, SplashScreen} from "./routes";

export default createAppContainer(
    createSwitchNavigator({
        SplashScreen,
        ItemScreens
    }))
