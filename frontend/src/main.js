import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import 'ol/ol.css'
import './assets/style.css'
import { installApiAuth } from './services/api'
import { useAuthStore } from './store/auth'

const pinia = createPinia()
const app = createApp(App)
app.use(pinia)
const authStore = useAuthStore(pinia)
authStore.restoreSession()
installApiAuth(authStore)
app.mount('#app')
