<template>
  <section class="auth-shell" data-testid="auth-overlay">
    <div class="auth-card">
      <div class="auth-eyebrow">{{ t('auth.eyebrow') }}</div>
      <h1>{{ t('auth.title') }}</h1>
      <p class="auth-copy">{{ t('auth.subtitle') }}</p>

      <div v-if="orgHint" class="auth-bootstrap">
        <div>{{ t('auth.bootstrapEmail') }}: <strong>{{ effectiveEmailHint }}</strong></div>
        <div>{{ t('auth.organization') }}: <strong>{{ orgHint }}</strong></div>
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label>
          <span>{{ t('auth.email') }}</span>
          <input v-model.trim="email" data-testid="auth-email" type="email" autocomplete="username" required />
        </label>
        <label>
          <span>{{ t('auth.password') }}</span>
          <input v-model="password" data-testid="auth-password" type="password" autocomplete="current-password" required />
        </label>
        <label>
          <span>{{ t('auth.organizationSlug') }}</span>
          <input
            v-model.trim="organizationSlug"
            data-testid="auth-org"
            type="text"
            autocomplete="organization"
            placeholder="default-organization"
          />
        </label>

        <button class="auth-submit" data-testid="auth-submit" :disabled="auth.isLoading">
          {{ auth.isLoading ? t('auth.signingIn') : t('auth.signIn') }}
        </button>
      </form>

      <p v-if="auth.error" class="auth-error">{{ auth.error }}</p>
      <p class="auth-hint">{{ t('auth.hint') }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

import { useAuthStore } from '../store/auth'
import { t } from '../utils/i18n'

const props = defineProps({
  initialEmail: {
    type: String,
    default: '',
  },
  initialOrganizationSlug: {
    type: String,
    default: '',
  },
  initialOrganizationName: {
    type: String,
    default: '',
  },
})

const auth = useAuthStore()
const email = ref('')
const password = ref('')
const organizationSlug = ref('')
const orgHint = computed(() => {
  const value = props.initialOrganizationName || props.initialOrganizationSlug || ''
  if (value === 'Default Organization') {
    return t('auth.defaultOrganizationName')
  }
  return value
})
const effectiveEmailHint = computed(() => props.initialEmail || 'admin@local')

watch(
  () => props.initialEmail,
  (value) => {
    if (!email.value && value) {
      email.value = value
    }
  },
  { immediate: true }
)

watch(
  () => props.initialOrganizationSlug,
  (value) => {
    if (!organizationSlug.value && value) {
      organizationSlug.value = value
    }
  },
  { immediate: true }
)

async function submit() {
  try {
    await auth.login({
      email: email.value,
      password: password.value,
      organizationSlug: organizationSlug.value,
    })
  } catch {
    // Error is already handled by the auth store.
  }
}
</script>

<style scoped>
.auth-shell {
  position: absolute;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 24px;
  background:
    linear-gradient(135deg, rgba(18, 43, 33, 0.88), rgba(11, 22, 44, 0.78)),
    radial-gradient(circle at top right, rgba(227, 177, 65, 0.18), transparent 34%);
  backdrop-filter: blur(8px);
}

.auth-card {
  width: min(460px, 100%);
  padding: 28px;
  border: 2px solid rgba(247, 239, 222, 0.34);
  background: rgba(17, 23, 28, 0.84);
  color: #f6f1e4;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
}

.auth-eyebrow {
  margin-bottom: 10px;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #d8c27a;
}

h1 {
  margin: 0 0 10px;
  font-size: 28px;
  line-height: 1.05;
}

.auth-copy,
.auth-hint,
.auth-error {
  margin: 0;
  font-size: 14px;
  line-height: 1.45;
}

.auth-bootstrap {
  margin-top: 14px;
  padding: 10px 12px;
  border: 1px solid rgba(246, 241, 228, 0.18);
  background: rgba(255, 255, 255, 0.05);
  font-size: 12px;
  line-height: 1.5;
}

.auth-form {
  display: grid;
  gap: 14px;
  margin: 20px 0 14px;
}

label {
  display: grid;
  gap: 6px;
}

label span {
  font-size: 12px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgba(246, 241, 228, 0.76);
}

input {
  border: 1px solid rgba(246, 241, 228, 0.24);
  background: rgba(255, 255, 255, 0.07);
  color: #f6f1e4;
  padding: 12px 14px;
  font: inherit;
}

input::placeholder {
  color: rgba(246, 241, 228, 0.38);
}

.auth-submit {
  margin-top: 4px;
  border: 0;
  padding: 13px 16px;
  font: inherit;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #12261e;
  background: linear-gradient(135deg, #d9c471, #f5e9bf);
  cursor: pointer;
}

.auth-submit:disabled {
  cursor: wait;
  opacity: 0.7;
}

.auth-error {
  color: #ffb5ab;
}

.auth-hint {
  margin-top: 12px;
  color: rgba(246, 241, 228, 0.68);
}
</style>
