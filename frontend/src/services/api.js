import axios from 'axios'

export const API_BASE = '/api/v1'

const api = axios.create()

let authStoreRef = null
let requestInterceptorId = null
let responseInterceptorId = null
let refreshPromise = null

export function installApiAuth(authStore) {
  authStoreRef = authStore

  if (requestInterceptorId !== null) {
    api.interceptors.request.eject(requestInterceptorId)
  }
  if (responseInterceptorId !== null) {
    api.interceptors.response.eject(responseInterceptorId)
  }

  requestInterceptorId = api.interceptors.request.use((config) => {
    const token = authStoreRef?.accessToken
    if (token) {
      config.headers = config.headers || {}
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  })

  responseInterceptorId = api.interceptors.response.use(
    (response) => response,
    async (error) => {
      const originalConfig = error?.config
      const status = error?.response?.status
      const requestUrl = String(originalConfig?.url || '')
      const isAuthRoute = requestUrl.includes('/auth/login') || requestUrl.includes('/auth/refresh')

      if (
        !authStoreRef ||
        status !== 401 ||
        !originalConfig ||
        originalConfig.__retried ||
        isAuthRoute ||
        !authStoreRef.refreshToken
      ) {
        return Promise.reject(error)
      }

      originalConfig.__retried = true

      if (!refreshPromise) {
        refreshPromise = authStoreRef.refreshSession().finally(() => {
          refreshPromise = null
        })
      }

      try {
        await refreshPromise
        originalConfig.headers = originalConfig.headers || {}
        originalConfig.headers.Authorization = `Bearer ${authStoreRef.accessToken}`
        return api(originalConfig)
      } catch (refreshError) {
        authStoreRef.clearSession()
        return Promise.reject(refreshError)
      }
    }
  )
}

export default api
