import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

import { API_BASE } from '../services/api'

const ACCESS_TOKEN_KEY = 'agromap-auth-access-token'
const REFRESH_TOKEN_KEY = 'agromap-auth-refresh-token'
const AUTH_USER_KEY = 'agromap-auth-user'

const authClient = axios.create()

export const useAuthStore = defineStore('auth', () => {
  const accessToken = ref('')
  const refreshToken = ref('')
  const user = ref(null)
  const isLoading = ref(false)
  const restored = ref(false)
  const error = ref('')

  const isAuthenticated = computed(() => Boolean(accessToken.value && refreshToken.value))

  function persistSession() {
    if (typeof window === 'undefined') {
      return
    }
    window.localStorage.setItem(ACCESS_TOKEN_KEY, accessToken.value)
    window.localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken.value)
    window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user.value || null))
  }

  function clearSession() {
    accessToken.value = ''
    refreshToken.value = ''
    user.value = null
    error.value = ''
    if (typeof window === 'undefined') {
      return
    }
    window.localStorage.removeItem(ACCESS_TOKEN_KEY)
    window.localStorage.removeItem(REFRESH_TOKEN_KEY)
    window.localStorage.removeItem(AUTH_USER_KEY)
  }

  function restoreSession() {
    if (typeof window !== 'undefined') {
      accessToken.value = window.localStorage.getItem(ACCESS_TOKEN_KEY) || ''
      refreshToken.value = window.localStorage.getItem(REFRESH_TOKEN_KEY) || ''
      const rawUser = window.localStorage.getItem(AUTH_USER_KEY)
      if (rawUser) {
        try {
          user.value = JSON.parse(rawUser)
        } catch {
          user.value = null
        }
      }
    }
    restored.value = true
  }

  function _applyTokenPayload(payload) {
    accessToken.value = payload.access_token
    refreshToken.value = payload.refresh_token
    user.value = payload.user
    error.value = ''
    persistSession()
  }

  async function login({ email, password, organizationSlug }) {
    isLoading.value = true
    error.value = ''
    try {
      const response = await authClient.post(`${API_BASE}/auth/login`, {
        email,
        password,
        organization_slug: organizationSlug || null,
      })
      _applyTokenPayload(response.data)
      return response.data
    } catch (requestError) {
      error.value =
        requestError?.response?.data?.detail || requestError?.message || 'Не удалось выполнить вход'
      throw requestError
    } finally {
      isLoading.value = false
    }
  }

  async function refreshSession() {
    if (!refreshToken.value) {
      throw new Error('Missing refresh token')
    }
    const response = await authClient.post(`${API_BASE}/auth/refresh`, {
      refresh_token: refreshToken.value,
    })
    _applyTokenPayload(response.data)
    return response.data
  }

  async function logout() {
    try {
      if (refreshToken.value) {
        await authClient.post(
          `${API_BASE}/auth/logout`,
          { refresh_token: refreshToken.value },
          {
            headers: accessToken.value ? { Authorization: `Bearer ${accessToken.value}` } : {},
          }
        )
      }
    } catch {
      // Logout is best-effort.
    } finally {
      clearSession()
    }
  }

  return {
    accessToken,
    refreshToken,
    user,
    isLoading,
    restored,
    error,
    isAuthenticated,
    restoreSession,
    clearSession,
    login,
    refreshSession,
    logout,
  }
})
