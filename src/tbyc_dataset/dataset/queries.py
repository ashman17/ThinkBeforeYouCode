ISSUE_LIST_QUERY = """
query IssueList(
  $owner: String!,
  $name: String!,
  $cursor: String,
  $pageSize: Int!,
  $states: [IssueState!]
) {
  repository(owner: $owner, name: $name) {
    issues(
      first: $pageSize,
      after: $cursor,
      states: $states,
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        state
        stateReason
        createdAt
        closedAt
        url
        comments {
          totalCount
        }
      }
    }
  }
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""


ISSUE_DETAIL_QUERY = """
query IssueDetail(
  $owner: String!,
  $name: String!,
  $number: Int!,
  $commentsPageSize: Int!,
  $commentsCursor: String,
  $timelinePageSize: Int!,
  $timelineCursor: String
) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      id
      number
      title
      body
      url
      state
      stateReason
      createdAt
      closedAt
      authorAssociation
      author {
        __typename
        login
      }
      labels(first: 100) {
        nodes {
          name
        }
      }
      comments(first: $commentsPageSize, after: $commentsCursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          url
          body
          createdAt
          authorAssociation
          author {
            __typename
            login
          }
        }
      }
      timelineItems(
        first: $timelinePageSize,
        after: $timelineCursor,
        itemTypes: [
          CLOSED_EVENT,
          REOPENED_EVENT,
          CROSS_REFERENCED_EVENT,
          LABELED_EVENT,
          UNLABELED_EVENT
        ]
      ) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          __typename
          ... on ClosedEvent {
            createdAt
            stateReason
            actor {
              __typename
              login
            }
          }
          ... on ReopenedEvent {
            createdAt
            actor {
              __typename
              login
            }
          }
          ... on CrossReferencedEvent {
            createdAt
            willCloseTarget
            actor {
              __typename
              login
            }
            source {
              __typename
              ... on PullRequest {
                number
                url
                title
                state
                merged
                mergedAt
              }
            }
          }
          ... on LabeledEvent {
            createdAt
            actor {
              __typename
              login
            }
            label {
              name
            }
          }
          ... on UnlabeledEvent {
            createdAt
            actor {
              __typename
              login
            }
            label {
              name
            }
          }
        }
      }
    }
  }
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""
