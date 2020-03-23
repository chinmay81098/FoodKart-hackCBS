import React from "react";
import Constants from 'expo-constants';
import {LinearGradient} from 'expo-linear-gradient'
import {Animated, TextInput, View, Text, StyleSheet, Image, KeyboardAvoidingView, TouchableOpacity} from "react-native"

type ButtonProps = {
    title: string
    backgroundColor: string
    color: string,
    onPress: () => void
}

type TFormFields = {
    title: string,
    secureTextEntry: boolean
}

let formFields: TFormFields[] = [
    {
        title: "Email",
        secureTextEntry: false
    },
    {
        title: "Password",
        secureTextEntry: true
    },
]

let payloadObjet = {
    email: "",
    password: "",
    confirmPassword: ""
};


// @ts-ignore
function FormField({title, secureTextEntry, onChangeText}) {
    return (
        <View>
            <View>
                <Text style={{color: "#303f9f"}}>
                    {title}
                </Text>
            </View>
            <TextInput
                {...{onChangeText, secureTextEntry}}
                blurOnSubmit={true}
                clearTextOnFocus={true}
                style={[{
                    borderBottomWidth: 1,
                    marginBottom: 7.5,
                    borderBottomColor: "#303f9f"
                }]}
            />
        </View>
    )
}

const FormWrapper = React.memo(({children}) => (
    <View style={[s.flex1, s.formWrapper]}>
        {children}
    </View>
));
const ViewOne = React.memo(() => (
    <View style={[s.center, s.flex1, s.alignCenter]}>
        <Image source={{uri: 'https://cdn.dribbble.com/users/4598/screenshots/3706567/flavors.jpg'}}
               style={{width: 100, height: 100, resizeMode: "contain"}}/>
        <Text style={{color: "#0A10C8", textAlign: "center", fontSize: 22, fontWeight: "bold"}}> Food
            Kart </Text>
    </View>
));

const LoginSigninButton = (({title, backgroundColor, color, onPress}: ButtonProps) => (
    <View style={[s.center, s.alignCenter, {},]}>
        <TouchableOpacity style={{
            borderRadius: 10,
            margin: 7.5,
            padding: 10,
            backgroundColor
        }} {...{onPress}}>
            <Text style={{color}}>{title}</Text>
        </TouchableOpacity>
    </View>
));


export default function AuthScreen(props) {
    const [isToggle, toggle] = React.useState(false);
    const [_formFields, setFormFields] = React.useState([...formFields]);
    const [isLogin, setLogin] = React.useState(true)
    const [cred, setEmail] = React.useState("")
    React.useEffect(() => {
        if (_formFields.length === 3) {

        }
    }, [isToggle])


    function onChangeText(text, key) {
        if (key === 0) {
            setEmail(text)
        }

        if (key === 1) {
            payloadObjet.password = text
        }

        if (key === 2) {
            payloadObjet.confirmPassword = text
        }

    }

    function callAuthentication(title) {
        const {email, password, confirmPassword} = payloadObjet
        if(password !== confirmPassword){
            alert("password not match")
        }    else {
            props.navigation.navigate("ItemScreens")
        }

    }

    return (
        <KeyboardAvoidingView style={[s.flex1, {marginTop: Constants.statusBarHeight}]} behavior="padding" enabled
                              contentContainerStyle={{flexGrow: 1}}>
            <LinearGradient style={s.flex1} colors={["#FFFEFF", "#D7FFFE"]}>
                <ViewOne/>
                <FormWrapper>
                    <View style={[s.flex2, s.center]}>
                        {
                            _formFields.map((props: TFormFields, key: number) => (
                                <FormField {...props} key={key}
                                           onChangeText={(text) => onChangeText(text, key)}/>
                            ))
                        }
                    </View>
                </FormWrapper>
                <LoginSigninButton
                    onPress={() => {
                        if (isLogin) {
                            callAuthentication("login")
                        } else {
                            callAuthentication("signin")
                        }
                    }}
                    title={isLogin ? "Login" : "Signup"}
                    backgroundColor={"#303f9f"}
                    color={"white"}
                />
                <LoginSigninButton
                    onPress={() => {
                        if (_formFields.length === 2) {
                            const temp = [
                                ..._formFields,
                                {
                                    title: "Confirm Password",
                                    secureTextEntry: true
                                },
                            ]
                            setFormFields(temp)
                        } else {
                            setFormFields(formFields)
                        }
                        setLogin((value) => !value)

                    }}
                    title={isLogin ? "Signup" : "Login"}
                    backgroundColor={"white"}
                    color={"#303f9f"}
                />

            </LinearGradient>
        </KeyboardAvoidingView>
    )
}

const s = StyleSheet.create({
    flex1: {
        flex: 1,
    },
    flex2: {
        flex: 2,
    },
    center: {
        justifyContent: "center",
    },
    alignCenter: {
        alignItems: "center"
    },
    formWrapper: {
        padding: 20,
    }
});
